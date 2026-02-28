
import PyNvVideoCodec as nvc
import inspect

# Get the Enum
CodecEnum = getattr(nvc, 'CudaVideoCodec', getattr(nvc, 'cudaVideoCodec', None))
print(f"CodecEnum found: {CodecEnum}")

print("\nCreateDecoder Signature:")
if hasattr(nvc, 'CreateDecoder'):
    try:
        print(inspect.signature(nvc.CreateDecoder))
    except ValueError:
        print(nvc.CreateDecoder.__doc__)

print("\nAttempting CreateDecoder:")
try:
    decoder = nvc.CreateDecoder(
        gpuid=0,
        codec=CodecEnum.H264,
        usedevicememory=True
    )
    print("SUCCESS: Created Decoder")
except Exception as e:
    print(f"FAILED: {type(e)} {e}")
