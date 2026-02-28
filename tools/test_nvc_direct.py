
import sys
import PyNvVideoCodec as nvc

path = "/workspace/input/test_video.mp4"
print(f"Testing library version: {nvc.__version__ if hasattr(nvc, '__version__') else 'unknown'}")

print(f"\nTest 1: CreateDemuxer(str) with {path}")
try:
    demuxer = nvc.CreateDemuxer(path)
    print("SUCCESS: Created Demuxer with string")
    print(f"Codec: {demuxer.Codec()}")
except Exception as e:
    print(f"FAILED: {type(e)} {e}")

print(f"\nTest 2: CreateDemuxer(bytes) with {path.encode('utf-8')}")
try:
    demuxer = nvc.CreateDemuxer(path.encode('utf-8'))
    print("SUCCESS: Created Demuxer with bytes")
    print(f"Codec: {demuxer.Codec()}")
except Exception as e:
    print(f"FAILED: {type(e)} {e}")

print(f"\nTest 3: CreateDemuxer(filename=str)")
try:
    demuxer = nvc.CreateDemuxer(filename=path)
    print("SUCCESS: Created Demuxer with filename kwarg")
    print("Demuxer Attributes:")
    print(dir(demuxer))
except Exception as e:
    print(f"FAILED: {type(e)} {e}")

print("\nDone.")
