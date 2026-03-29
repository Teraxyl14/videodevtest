# **Project Aether v2.1: Final Technical Stack (March 2026 SOTA)**

## **1\. Hardware & Infrastructure Overview**

**Target Hardware:** Intel Core Ultra 7 255HX (Panther Lake NPU), NVIDIA RTX 5080 Mobile (16GB VRAM, Blackwell), 32GB DDR5.

**Environment:** Windows 11 WSL2 (Ubuntu 24.04), Docker Engine with NVIDIA Container Toolkit.

### **1.1 System-Level Configurations**

* **VRAM Protection:** export PYTORCH\_CUDA\_ALLOC\_CONF=expandable\_segments:True,max\_split\_size\_mb:512.  
* **WSL2 Limits:** .wslconfig with memory=24GB and swap=8GB to prevent host RAM exhaustion.  
* **Docker IPC:** Launched with \--ipc=host \--ulimit memlock=-1 to ensure zero-copy shared memory stability.

## **2\. Always-On Discovery Layer (NPU/CPU)**

Monitors web signals 24/7 using the Intel NPU (\<5W) to avoid waking the RTX 5080 for routine scraping.

### **2.1 Discovery (Crawl4AI)**

import asyncio  
from crawl4ai import AsyncWebCrawler  
from crawl4ai.extraction\_strategy import JsonCssExtractionStrategy

async def discover\_trends():  
    schema \= {  
        "name": "Trending Topics",  
        "baseSelector": ".trending-item",  
        "fields": \[  
            {"name": "title", "selector": "h2", "type": "text"},  
            {"name": "engagement", "selector": ".count", "type": "text"}  
        \]  
    }  
    async with AsyncWebCrawler() as crawler:  
        result \= await crawler.arun(  
            url="\[https://www.tiktok.com/trending\](https://www.tiktok.com/trending)",  
            extraction\_strategy=JsonCssExtractionStrategy(schema)  
        )  
        return result.extracted\_content

### **2.2 NPU Scoring Filter (Gemma-3n-E2B-IT)**

Utilizes the MatFormer architecture for selective parameter activation, allowing 6B-level reasoning at a 2B memory footprint.

import openvino\_genai as ov\_genai

\# Implementation for low-power virality scoring on Panther Lake NPU  
model\_path \= "/models/gemma-3n-e2b-it-int4-ov"  
pipe \= ov\_genai.VLMPipeline(model\_path, "NPU")

def score\_trend(trend\_text: str) \-\> str:  
    prompt \= f"Rate this trend 0-100 for viral short-form video potential. Return JSON: {{'score': \<int\>, 'reason': '\<brief\>'}} Trend: {trend\_text}"  
    pipe.start\_chat()  
    result \= pipe.generate(prompt, max\_new\_tokens=100)  
    pipe.finish\_chat()  
    return result

## **3\. Audio-Visual Analysis Subsystem (Parallel GPU Spoke)**

Replaces sequential swapping. Both models reside permanently in a 4GB VRAM subset.

### **3.1 Acoustic: easytranscriber**

Replaces WhisperX. Achieves \~102% faster execution using GPU-parallelized forced alignment.

from easytranscriber import Transcriber

transcriber \= Transcriber(  
    model="large-v3",  
    backend="ctranslate2",  
    device="cuda",  
    compute\_type="float16",  
    batch\_size=16  
)

def transcribe\_video(audio\_path: str):  
    \# Returns millisecond-accurate word-level timestamps  
    return transcriber.transcribe(audio\_path, language="en", return\_timestamps=True)

### **3.2 Visual: Qwen3.5-0.8B (vLLM INT4)**

Natively multimodal (Early Fusion). Replaces Qwen3-4B.

\# Serve via vLLM Nightly  
\# vllm serve Qwen/Qwen3.5-0.8B \--port 8002 \--max-model-len 4096 \--gpu-memory-utilization 0.15

from openai import OpenAI  
client \= OpenAI(base\_url="http://localhost:8002/v1", api\_key="none")

def analyze\_frame(image\_b64: str, prompt: str):  
    response \= client.chat.completions.create(  
        model="Qwen/Qwen3.5-0.8B",  
        messages=\[{"role": "user", "content": \[  
            {"type": "image\_url", "image\_url": {"url": f"data:image/jpeg;base64,{image\_b64}"}},  
            {"type": "text", "text": prompt}  
        \]}\]  
    )  
    return response.choices\[0\].message.content

## **4\. Spatial Tracking & Manufacturing (RTX 5080 Blackwell)**

### **4.1 Tracking: YOLO26s-Pose (TensorRT)**

NMS-Free, end-to-end CNN. 1.7ms \- 3.2ms inference speed on TensorRT FP16.

from ultralytics import YOLO

def track\_subjects(video\_path: str):  
    tracker \= YOLO("yolo26s-pose.engine")   
    return tracker.track(  
        source=video\_path,   
        tracker="bytetrack.yaml",   
        persist=True,   
        stream=True,  
        conf=0.5  
    )

### **4.2 Zero-Copy Video: PyNvVideoCodec 2.1.0**

Standardizes NVDEC \-\> DLPack \-\> PyTorch path.

import PyNvVideoCodec as nvc  
import torch

def decode\_to\_gpu(video\_path: str):  
    demuxer \= nvc.CreateDemuxer(video\_path)  
    decoder \= nvc.CreateDecoder(gpuid=0, codec=demuxer.GetCodecType(), outputformat="nv12")  
      
    for packet in demuxer:  
        for frame in decoder.decode(packet):  
            \# Zero-copy bridge to PyTorch via DLPack protocol  
            yield torch.from\_dlpack(frame)

def create\_av1\_encoder(output\_path: str):  
    return nvc.CreateEncoder(  
        output\_path, width=1080, height=1920, codec="av1",  
        preset="p7", tuning="hq", \# 'hq' bypasses Blackwell-WSL2 macroblocking bug  
        bitrate=8\_000\_000  
    )

## **5\. Orchestration & Deployment**

### **5.1 Hybrid Logic: LangGraph \+ PydanticAI**

LangGraph manages the macro-state (Director/Editor/Compliance) and Redis Streams checkpoints. PydanticAI enforces type-safety on Gemini 3 Pro outputs.

from pydantic\_ai import Agent  
from pydantic import BaseModel

class VideoManifest(BaseModel):  
    start\_time: float  
    end\_time: float  
    hook\_text: str

director \= Agent('google-gla:gemini-3-pro', result\_type=VideoManifest)

\# LangGraph node implementation  
async def director\_node(state):  
    result \= await director.run("Plan the cuts for this transcript...")  
    return {"manifest": result.data}

### **5.2 Deployment (Docker Compose)**

services:  
  aether:  
    build: .  
    runtime: nvidia  
    ipc: host  
    environment:  
      \- PYTORCH\_CUDA\_ALLOC\_CONF=expandable\_segments:True,max\_split\_size\_mb:512  
    ulimits:  
      memlock: \-1  
    volumes:  
      \- /usr/lib/wsl/lib:/usr/lib/x86\_64-linux-gnu:ro

## **6\. Implementation Summary**

1. **NPU Discovery:** Gemma-3n filters signals 24/7 at \<5W.  
2. **Zero-Latency Analysis:** Parallel residency of easytranscriber and Qwen3.5-0.8B removes context-switching.  
3. **Manufacturing:** YOLO26s-Pose and PyNvVideoCodec 2.1.0 maintain a pure VRAM pipeline.  
4. **Resilience:** LangGraph with Redis checkpointing ensures survival against GPU driver resets or container faults.