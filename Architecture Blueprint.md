# **Autonomous AI Video Agent (v2.0) \- Architecture & Model Migration Blueprint**

## **1\. Core Infrastructure & Memory Orchestration**

The system utilizes a Hub-and-Spoke architecture to strictly manage the 16GB VRAM constraint of the RTX 5080 Mobile. The following structural configurations must be applied to the Windows 11 host and the Docker runtime.

### **1.1 WSL2 & Docker IPC Boundaries**

* **WSL2 Memory Ceiling:** Configure the host .wslconfig to hard-limit memory to 24GB and swap to 8GB to prevent vmmem runaway allocation and host system crashes during Python model process iteration.  
* **IPC Namespace:** The Docker container must be initiated with \--ipc=host (or \--shm-size=16gb) to bypass the default 64MB /dev/shm limit, preventing fatal bus errors when passing 100MB shared CUDA buffers.  
* **Pinned Memory Integrity:** Execute Docker with \--ulimit memlock=-1 to prevent page-outs of pinned memory blocks. Limit any host-pinned memory allocations to under 500MB to bypass the undocumented NVIDIA Container Toolkit WSL2 failure limit.

### **1.2 PyTorch VRAM Fragmentation Mitigation**

* **CUDA Allocator:** Export the environment variable PYTORCH\_CUDA\_ALLOC\_CONF=expandable\_segments:True,max\_split\_size\_mb:512. This forces PyTorch to use CUDA Virtual Memory Management to map discontiguous physical pages into contiguous virtual space, preventing artificial OOM errors when hot-swapping models with different tensor geometries.

### **1.3 The "Locker" Process Persistence & IPC Routing**

* **IPC Handle Volatility:** NVIDIA Triton Inference Server is incompatible with WSL2 WDDM paravirtualization for shared GPU memory. The system must retain the custom Python "Locker" process.  
* **Implementation:** The Locker process allocates the 100MB zero-copy buffer and remains alive in an infinite wait state. Ephemeral spoke processes access this memory via torch.multiprocessing queues utilizing DLPack.

import os

\# 1\. Enforce strict VRAM fragmentation management for the RTX 5080 Mobile  
\# This must be executed prior to importing torch to alter the caching allocator.  
os.environ\["PYTORCH\_CUDA\_ALLOC\_CONF"\] \= "expandable\_segments:True,max\_split\_size\_mb:512"

import torch  
import torch.multiprocessing as mp

def initialize\_persistent\_hub\_buffer(buffer\_size\_mb: int \= 100\) \-\> tuple:  
    """  
    Allocates a persistent, contiguous block of VRAM in the Hub process.  
    Generates the 64-byte IPC handle required for zero-copy Spoke access.  
    """  
    buffer\_bytes \= buffer\_size\_mb \* 1024 \* 1024  
    shared\_cuda\_tensor \= torch.empty(buffer\_bytes, dtype=torch.uint8, device='cuda')

    \# Extract the CUDA IPC handle via PyTorch reductions  
    ipc\_handle \= torch.multiprocessing.reductions.reduce\_tensor(shared\_cuda\_tensor)  
      
    return shared\_cuda\_tensor, ipc\_handle

def ephemeral\_spoke\_ingestion\_protocol(ipc\_handle: tuple) \-\> torch.Tensor:  
    """  
    Executed strictly within the isolated Spoke process.  
    Reconstructs the PyTorch tensor mapping to the exact physical VRAM  
    without initiating a host-to-device memory transfer.  
    """  
    rebuild\_function, rebuild\_args \= ipc\_handle\[0\], ipc\_handle\[1\]  
    spoke\_tensor \= rebuild\_function(\*rebuild\_args)  
      
    return spoke\_tensor

## **2\. Video Processing & Hardware Encoding**

The reliance on FFmpeg stream copying (-c copy) for 50-millisecond precise cuts is deprecated due to inter-frame (GOP) compression snapping.

* **Hardware Re-encoding:** The 60-second chunking logic utilizes a full hardware decode/re-encode pass using the Blackwell NVENC AV1 encoder.  
* **Bug Mitigation:** Use \-tune hq exclusively. Do *not* use \-tune uhq coupled with 10-bit depth, as it triggers a known Blackwell architectural bug causing macroblocking artifacts.

def generate\_blackwell\_av1\_command(word\_segment: dict, source\_video\_path: str, output\_video\_path: str) \-\> str:  
    """  
    Translates WhisperX millisecond timestamps into a frame-accurate,  
    hardware-accelerated FFmpeg cutting command utilizing the NVENC AV1 encoder.  
    """  
    start\_time\_float \= word\_segment.get("start")  
    end\_time\_float \= word\_segment.get("end")

    if start\_time\_float is None or end\_time\_float is None:  
        raise ValueError("Critical error: Word segment is missing wav2vec2 alignment timestamps.")

    duration\_float \= end\_time\_float \- start\_time\_float

    def format\_to\_ffmpeg\_timestamp(seconds: float) \-\> str:  
        hours \= int(seconds // 3600\)  
        minutes \= int((seconds % 3600\) // 60\)  
        secs \= int(seconds % 60\)  
        milliseconds \= int((seconds \- int(seconds)) \* 1000\)  
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{milliseconds:03d}"

    formatted\_start \= format\_to\_ffmpeg\_timestamp(start\_time\_float)  
    formatted\_duration \= format\_to\_ffmpeg\_timestamp(duration\_float)

    \# Note: \-ss is positioned prior to \-i to enable fast seeking  
    ffmpeg\_execution\_string \= (  
        f"ffmpeg \-hwaccel cuda \-hwaccel\_output\_format cuda \-ss {formatted\_start} \-t {formatted\_duration} \-i {source\_video\_path} "  
        f"-c:v av1\_nvenc \-preset p7 \-tune hq \-rc vbr \-cq 19 \-b:v 0 \-rc-lookahead 48 "  
        f"-c:a copy \-y {output\_video\_path}"  
    )

    return ffmpeg\_execution\_string

## **3\. Acoustic Subsystem Migration**

* **Deprecating:** Parakeet TDT 0.6B v2 and NeMo TitaNet. (Issues: coarse timestamping, phonetic hallucination, sequential processing bottleneck).  
* **New Architecture:** **WhisperX** utilizing **Distil-Whisper Large-v3** and **Pyannote 4.0**.  
* **Rationale:** Reduces VRAM footprint to \~3.0GB (FP16). Integrates wav2vec2 forced-alignment for millisecond-precise word-level timestamps directly required for the FFmpeg re-encode. Pyannote natively clusters speakers via agglomerative logic.

import whisperx  
import gc  
import torch

def execute\_audio\_alignment\_spoke(audio\_filepath: str, hf\_auth\_token: str) \-\> dict:  
    """  
    Ephemeral Spoke execution for WhisperX. Sequentially loads the ASR,  
    Forced Alignment, and Diarization models, clearing VRAM between stages.  
    """  
    execution\_device \= "cuda"  
    tensor\_precision \= "float16"

    \# Stage 1: Initial Transcription via Distil-Whisper Large-v3  
    asr\_pipeline \= whisperx.load\_model("distil-large-v3", execution\_device, compute\_type=tensor\_precision)  
    raw\_audio\_waveform \= whisperx.load\_audio(audio\_filepath)  
    transcription\_result \= asr\_pipeline.transcribe(raw\_audio\_waveform, batch\_size=16)

    \# Aggressively reclaim VRAM prior to loading the phoneme-based model  
    del asr\_pipeline  
    gc.collect()  
    torch.cuda.empty\_cache()

    \# Stage 2: Forced Alignment via wav2vec2  
    alignment\_model, alignment\_metadata \= whisperx.load\_align\_model(  
        language\_code=transcription\_result\["language"\],  
        device=execution\_device  
    )  
      
    aligned\_segments \= whisperx.align(  
        transcription\_result\["segments"\],  
        alignment\_model,  
        alignment\_metadata,  
        raw\_audio\_waveform,  
        execution\_device,  
        return\_char\_alignments=False  
    )

    del alignment\_model  
    gc.collect()  
    torch.cuda.empty\_cache()

    \# Stage 3: Speaker Diarization via Pyannote 4.0  
    diarization\_pipeline \= whisperx.DiarizationPipeline(  
        use\_auth\_token=hf\_auth\_token,  
        device=execution\_device  
    )  
      
    speaker\_segments \= diarization\_pipeline(audio\_filepath, min\_speakers=1, max\_speakers=5)  
    fully\_attributed\_result \= whisperx.assign\_word\_speakers(speaker\_segments, aligned\_segments)

    del diarization\_pipeline  
    gc.collect()  
    torch.cuda.empty\_cache()

    return fully\_attributed\_result

## **4\. Vision-Language Cognition (VLM)**

* **Deprecating:** Qwen2.5-VL-7B-Instruct in FP8. (Issue: KV cache prefilling on temporal data saturates the 16GB bus, causing OS paging and massive TTFT latency).  
* **New Architecture:** **Qwen3-4B** in INT4 quantization via vLLM.  
* **Rationale:** Operates strictly under a 3.8GB VRAM footprint with a 262k context window. Features a dynamic dual-mode architecture allowing the orchestrator to toggle enable\_thinking based on the frame's heuristic value.

from vllm import LLM, SamplingParams  
import torch  
import gc

def initialize\_ephemeral\_qwen3\_spoke() \-\> LLM:  
    """  
    Initializes Qwen3-4B utilizing INT4 AWQ quantization via the vLLM engine.  
    Configured specifically for aggressive VRAM management (Caps at \~5.6GB).  
    """  
    reasoning\_engine \= LLM(  
        model="Qwen/Qwen3-4B-Instruct-AWQ",  
        quantization="awq",  
        tensor\_parallel\_size=1,  
        gpu\_memory\_utilization=0.35,   
        enforce\_eager=True,           \# Disables CUDA graph caching to conserve VRAM overhead  
        enable\_reasoning=True         \# Prepares backend to parse CoT \<think\> output  
    )  
    return reasoning\_engine

def execute\_dual\_mode\_reasoning(reasoning\_engine: LLM, contextual\_prompt: str, trigger\_deep\_thought: bool) \-\> str:  
    """  
    Executes inference dynamically, explicitly toggling the Chain-of-Thought  
    cognitive graph based on the complexity of the analytical task.  
    """  
    conversation\_history \= \[{"role": "user", "content": contextual\_prompt}\]

    if trigger\_deep\_thought:  
        \# Thinking Mode Configuration (High latency, deep CoT generation)  
        inference\_params \= SamplingParams(temperature=0.6, top\_p=0.95, max\_tokens=4096)  
        template\_kwargs \= {"enable\_thinking": True}  
    else:  
        \# Non-Thinking Mode Configuration (Sub-second latency, deterministic)  
        inference\_params \= SamplingParams(temperature=0.0, top\_p=1.0, max\_tokens=256)  
        template\_kwargs \= {"enable\_thinking": False}

    generation\_output \= reasoning\_engine.chat(  
        messages=conversation\_history,  
        sampling\_params=inference\_params,  
        chat\_template\_kwargs=template\_kwargs  
    )

    return generation\_output\[0\].outputs\[0\].text

def terminate\_qwen3\_spoke(reasoning\_engine: LLM):  
    """Forcibly unmaps the PagedAttention blocks and returns VRAM to OS."""  
    del reasoning\_engine  
    gc.collect()  
    torch.cuda.empty\_cache()

## **5\. Spatial Geometry & Tracking**

* **Deprecating:** The dual-model approach of YOLOv12-Face Nano (attention-heavy) \+ YOLO26n-Pose.  
* **New Architecture:** **YOLO26s-Pose** (Single Pass).  
* **Rationale:** YOLO26 is NMS-free. A single forward pass inherently extracts the full-body bounding box AND the 17-point skeletal matrix. Drops VRAM overhead to \<200MB.

import torch  
import PyNvVideoCodec as nvc  
from ultralytics import YOLO

def initialize\_yolo26\_pose\_spoke() \-\> YOLO:  
    """Initializes the end-to-end NMS-free YOLO26s-Pose model."""  
    spatial\_model \= YOLO("yolo26s-pose.pt")  
    spatial\_model.to("cuda")  
    return spatial\_model

def execute\_zero\_copy\_ingestion(decoder\_instance: nvc.PyNvDecoder, spatial\_model: YOLO) \-\> list:  
    """  
    Decodes video frames directly into VRAM and routes them into YOLO26  
    utilizing DLPack, completely bypassing the PCIe bus and CPU.  
    """  
    batch\_processing\_size \= 16  
    hardware\_decoded\_frames \= decoder\_instance.get\_batch\_frames(batch\_processing\_size)

    if len(hardware\_decoded\_frames) \== 0:  
        return \[\]

    zero\_copy\_tensor\_batch \= \[\]

    for raw\_frame in hardware\_decoded\_frames:  
        \# Convert NVDEC frame to PyTorch tensor via DLPack mapping (O(1) operation)  
        frame\_tensor \= torch.from\_dlpack(raw\_frame)  
          
        \# YOLO26 requires channel-first layout (permute manipulates stride, not physical memory)  
        frame\_tensor \= frame\_tensor.permute(2, 0, 1).float() / 255.0  
        frame\_tensor \= frame\_tensor.unsqueeze(0)  
        zero\_copy\_tensor\_batch.append(frame\_tensor)

    final\_batch\_tensor \= torch.cat(zero\_copy\_tensor\_batch, dim=0)  
    prediction\_results \= spatial\_model.predict(final\_batch\_tensor, half=True, verbose=False)

    return prediction\_results

def extract\_tracking\_matrices(yolo\_prediction\_result) \-\> tuple:  
    """  
    Extracts the isolated full-body bounding box and the 5-point facial landmark  
    matrix from the unified YOLO26s-Pose tensor via virtual slicing.  
    """  
    if yolo\_prediction\_result.boxes is None or yolo\_prediction\_result.keypoints is None:  
        return None, None

    \# 1\. Isolate Bounding Box array for ByteTrack (Shape: N, 6\)  
    bbox\_tracking\_tensor \= yolo\_prediction\_result.boxes.xyxy

    \# 2\. Extract unified 17-point Keypoint Tensor  
    full\_anatomical\_keypoints \= yolo\_prediction\_result.keypoints.data

    \# Matrix Slice logic isolating Indices: 0 (Nose), 1 (L-Eye), 2 (R-Eye), 3 (L-Ear), 4 (R-Ear)  
    facial\_landmark\_tensor \= full\_anatomical\_keypoints\[:, 0:5, :\]

    return bbox\_tracking\_tensor, facial\_landmark\_tensor

## **6\. Cloud-Edge Cognitive Offloading (Gemini 3 Flash)**

* **Agentic Vision (Spatial Analysis):** Deprecate local bounding-box cropping for scene analysis. Send the raw, wide-angle DLPack/Tensor frame directly to the Gemini API with media\_resolution=HIGH and Agentic Vision tools enabled. This allows Google's backend to execute Python logic natively (Think-Act-Observe).  
* **Metadata Synthesis:** Route final scripts and SEO extraction to Gemini with thinking\_level=MINIMAL.

import os  
from google import genai  
from google.genai import types

def execute\_agentic\_vision\_delegation\_spoke(raw\_image\_bytes: bytes, spatial\_directive: str) \-\> str:  
    """  
    Offloads highly complex spatial mathematics to the Gemini 3 Flash API.  
    Utilizes the Agentic Vision framework to execute deterministic OpenCV  
    analysis on the Google backend, entirely bypassing the local RTX 5080 GPU.  
    """  
    \# Requires GOOGLE\_API\_KEY environment variable  
    cloud\_client \= genai.Client()

    multimodal\_image\_part \= types.Part.from\_bytes(  
        data=raw\_image\_bytes,  
        mime\_type="image/jpeg"  
    )

    request\_configuration \= types.GenerateContentConfig(  
        \# 1\. Enforce high-density pixel upload for backend cropping algorithms  
        media\_resolution=types.MediaResolution.MEDIA\_RESOLUTION\_HIGH,  
          
        \# 2\. Constrain the textual reasoning verbosity to reduce network latency  
        thinking\_config=types.ThinkingConfig(  
            thinking\_level=types.ThinkingLevel.MINIMAL  
        ),  
          
        \# 3\. Enable the Deterministic Code Execution Sandbox (Think-Act-Observe loop)  
        tools=\[types.Tool(code\_execution=types.ToolCodeExecution)\]  
    )

    api\_response \= cloud\_client.models.generate\_content(  
        model="gemini-3-flash-preview",  
        contents=\[multimodal\_image\_part, spatial\_directive\],  
        config=request\_configuration  
    )

    return api\_response.text  
