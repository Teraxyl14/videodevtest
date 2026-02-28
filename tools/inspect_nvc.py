
import logging
import PyNvVideoCodec as nvc

print("PyNvVideoCodec contents:")
print(dir(nvc))

try:
    print("\nAttempting to create generic demuxer:")
    if hasattr(nvc, 'CreateDemuxer'):
        print("Found CreateDemuxer")
        # Try to inspect signature
        import inspect
        try:
            sig = inspect.signature(nvc.CreateDemuxer)
            print(f"Signature: {sig}")
        except ValueError:
            print("Could not get signature (C extension)")
            print(nvc.CreateDemuxer.__doc__)
            
    else:
        print("CreateDemuxer NOT found")
except:
    pass
