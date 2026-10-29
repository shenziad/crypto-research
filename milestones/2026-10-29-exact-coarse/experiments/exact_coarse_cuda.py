"""GPU public compiler with an explicitly opened exact coarse classifier."""
import torch


def compile_public(runtime,h,classifier):
    t=runtime;s,e,a,c=[t[k] for k in ('starts','ends','slopes','intercepts')];vm=(1<<t['view'])-1;low=classifier&vm
    cx=(low[:,None]>s[None,:]).to(torch.int64)-(low[:,None]>((e+1)&vm)[None,:]).to(torch.int64)
    t.update(h=h,query=((low[:,None]-s[None,:]-1)&vm).contiguous(),
        c0=(cx*(a[None,:]*h[:,None]+c[None,:])).sum(1),c1=(cx*(-a[None,:])).sum(1),
        selector_c0=(cx*a[None,:]).sum(1),selector_c1=(cx*c[None,:]).sum(1))
    return t
