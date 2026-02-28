import torch
try:
    import PyNvVideoCodec as nvc
except ImportError:
    print("PyNvVideoCodec not found")
    exit(1)

def probe(video_path):
    print(f"Probing: {video_path}")
    nvDemux = nvc.CreateDemuxer(filename=video_path)
    try:
        codec = nvDemux.GetNvCodecId()
        print(f"Detected Codec ID: {codec}")
        nvDec = nvc.CreateDecoder(gpuid=0, codec=codec)
    except Exception as e:
        print(f"Decoder init failed: {e}")
        return
    except Exception as e:
        print(f"Decoder init failed: {e}")
        return
    
    for i, packet in enumerate(nvDemux):
        if i == 0:
            print("Packet Attributes:", dir(packet))
        
        # blindly try to decode
        surfaces = nvDec.Decode(packet)
        if surfaces:
             for surface in surfaces:
                 try:
                     import torch.utils.dlpack
                     # Try native conversion first
                     if hasattr(surface, 'nv12_to_rgb'):
                         print("Attempting nv12_to_rgb()...")
                         rgb_surf = surface.nv12_to_rgb()
                         tensor = torch.utils.dlpack.from_dlpack(rgb_surf)
                         print(f"RGB Tensor Shape: {tensor.shape}")
                         print(f"RGB Tensor Dtype: {tensor.dtype}")
                     else:
                         tensor = torch.utils.dlpack.from_dlpack(surface)
                         print(f"Raw Tensor Shape: {tensor.shape}")
                     return
                 except Exception as e:
                     print(f"Conversion/DLPack failed: {e}")
                     return

if __name__ == "__main__":
    # Use a dummy path or a real one if available. 
    # We will pass a known video path when running.
    import sys
    if len(sys.argv) > 1:
        probe(sys.argv[1])
    else:
        print("Usage: python test_nv_decode.py <video_path>")
