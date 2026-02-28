
import PyNvVideoCodec as nvc

print("NVC Directory:")
for x in dir(nvc):
    if "H264" in x or "Codec" in x or "cuda" in x.lower():
        print(f"  {x}")

print("\nDirect checks:")
try:
    print("nvc.cudaVideoCodec members:")
    for x in dir(nvc.cudaVideoCodec):
        if not x.startswith("_"):
            print(f"  {x}")
except Exception as e:
    print(f"Error inspecting enum: {e}")
