"""Fetch official Qwen0.5B assets to NEWSMY, pinned revision and LFS SHA256."""
import hashlib
import json
import time
import urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'models/Qwen2.5-0.5B'
REV='060db6499f32faf8b98477b0a26969ef7d8b9987'
MODEL_SHA='88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342'
MODEL_SIZE=988097824
FILES=['config.json','generation_config.json','tokenizer.json','tokenizer_config.json','merges.txt','vocab.json','LICENSE','model.safetensors']


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4<<20),b''):h.update(block)
    return h.hexdigest()


def main():
    DEST.mkdir(parents=True,exist_ok=True);rows=[]
    for name in FILES:
        path=DEST/name
        if path.exists():
            if name=='model.safetensors':assert path.stat().st_size==MODEL_SIZE and digest(path)==MODEL_SHA
        else:
            part=DEST/(name+'.part');offset=part.stat().st_size if part.exists() else 0
            url=f'https://huggingface.co/Qwen/Qwen2.5-0.5B/resolve/{REV}/{name}?download=true'
            req=urllib.request.Request(url,headers={'Range':f'bytes={offset}-'} if offset else {})
            with urllib.request.urlopen(req,timeout=45) as response:
                append=offset and response.status==206
                if not append:offset=0
                last=time.monotonic()
                with part.open('ab' if append else 'wb') as out:
                    for block in iter(lambda:response.read(4<<20),b''):
                        out.write(block);offset+=len(block)
                        if time.monotonic()-last>10:
                            print(name,offset,'bytes',flush=True);last=time.monotonic()
            if name=='model.safetensors':assert part.stat().st_size==MODEL_SIZE and digest(part)==MODEL_SHA
            part.rename(path)
        rows.append({'file':name,'bytes':path.stat().st_size,'sha256':digest(path)})
        print('verified',name,path.stat().st_size,flush=True)
    (DEST/'manifest.json').write_text(json.dumps({'model':'Qwen/Qwen2.5-0.5B','revision':REV,'official_lfs_sha256':MODEL_SHA,'files':rows},indent=2)+'\n')


if __name__=='__main__':main()
