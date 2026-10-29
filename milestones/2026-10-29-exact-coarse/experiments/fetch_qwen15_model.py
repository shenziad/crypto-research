"""Official Qwen1.5B assets pinned to the audited public API revision on NEWSMY."""
from pathlib import Path
import fetch_qwen_gate_model as fetch

fetch.DEST=fetch.ROOT/'models/Qwen2.5-1.5B'
fetch.REV='8faed761d45a263340a0528343f099c05c9a4323'
fetch.MODEL_SHA='a961db72e75d52b18e6b0c9d379e51a26973b233385e0e127fdda7d648aec796'
fetch.MODEL_SIZE=3087467144


def main():
    # Reuse digest and file-list helpers; verify this repository's own metadata.
    import hashlib
    import json
    import time
    import urllib.request
    dest=fetch.DEST;dest.mkdir(parents=True,exist_ok=True);rows=[]
    for name in fetch.FILES:
        path=dest/name
        if not path.exists():
            part=dest/(name+'.part');offset=part.stat().st_size if part.exists() else 0
            url=f'https://huggingface.co/Qwen/Qwen2.5-1.5B/resolve/{fetch.REV}/{name}?download=true'
            request=urllib.request.Request(url,headers={'Range':f'bytes={offset}-'} if offset else {})
            with urllib.request.urlopen(request,timeout=45) as response:
                append=bool(offset and response.status==206)
                if not append:offset=0
                last=time.monotonic()
                with part.open('ab' if append else 'wb') as out:
                    for block in iter(lambda:response.read(4<<20),b''):
                        out.write(block);offset+=len(block)
                        if time.monotonic()-last>10:print(name,offset,'bytes',flush=True);last=time.monotonic()
            part.rename(path)
        if name=='model.safetensors':assert path.stat().st_size==fetch.MODEL_SIZE and fetch.digest(path)==fetch.MODEL_SHA
        else:
            metadata=json.loads((dest/'official_api_metadata.json').read_text())
            entry=next(v for v in metadata['siblings'] if v['rfilename']==name)
            data=path.read_bytes();assert len(data)==entry['size']
            blob=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
            assert blob==entry['blobId']
        rows.append(dict(file=name,bytes=path.stat().st_size,sha256=fetch.digest(path)))
        print('verified',name,path.stat().st_size,flush=True)
    (dest/'manifest.json').write_text(json.dumps(dict(model='Qwen/Qwen2.5-1.5B',revision=fetch.REV,
        official_lfs_sha256=fetch.MODEL_SHA,files=rows),indent=2)+'\n')


if __name__=='__main__':main()
