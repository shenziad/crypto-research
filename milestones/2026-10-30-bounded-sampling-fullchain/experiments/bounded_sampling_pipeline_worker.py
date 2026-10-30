"""One logical party: secret alias -> private integer A row -> finite GC.

No intermediate reconstruction or online parent helper. Each invocation gets
only its own input file, socket, output directory and public shape/ports.
Existing passive OT/Yao/FABLE/SCI components, not new cryptographic primitives.
"""
import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
import sys
import time

import numpy as np

import alias_small_yao as yao
from finite_alias_table import build
from ristretto_ot import Channel,otn_send,otn_receive

ROOT=Path(__file__).resolve().parents[1]
N=65536;K=201;BATCH=256;P=44;ELL=16;R=8


def native(command,log):
    with log.open('xb') as out:
        subprocess.run(command,stdout=out,stderr=subprocess.STDOUT,check=True,timeout=180)


def main():
    if len(sys.argv)!=7:
        raise ValueError('role socketfd inputfile ownfolder fableport gcport')
    role=int(sys.argv[1]);assert role in (0,1)  # client=0, server=1
    private_file=Path(sys.argv[3]);folder=Path(sys.argv[4])
    fable_port=int(sys.argv[5]);gc_port=int(sys.argv[6])
    started=time.perf_counter()
    own=np.load(private_file)
    if role==0:
        x=own['x_integer'].astype(np.int64)
        assert x.shape==(128,)
        cut,alias,mass,represented,table_stats=build(own['q'],P)
        assert len(cut)==N and table_stats['padded_mass']==0
        np.savez_compressed(folder/'client_alias_audit.npz',cut=cut,alias=alias,mass=mass)
    else:
        ai=own['a_integer'].astype(np.int64);bi=own['b_integer'].astype(np.int64)
        assert ai.shape==(50304,8) and bi.shape==(8,128)
        assert np.abs(ai).max()<=65536 and np.abs(bi).max()<=65536
        table=np.zeros((N,R),dtype='<i4');table[:50304]=ai.astype('<i4')
        (folder/'owner_integer_table.bin').write_bytes(table.tobytes())
        table_stats=None
    own.close()
    prepared=time.perf_counter()
    sock=socket.socket(fileno=int(sys.argv[2]));sock.settimeout(180)
    channel=Channel(sock);circuit=yao.Circuit(ELL,P)
    index_shares=[];local_coins=[]
    for i in range(K):
        if role==0:
            kappa=secrets.randbits(ELL);tau=secrets.randbits(P+1)
            pi=secrets.randbits(ELL);us=secrets.randbits(P);rho=secrets.randbits(ELL)
            messages=[((int(cut[j^kappa])^tau)|((int(alias[j^kappa])^pi)<<(P+1))).to_bytes(8,'little')
                      for j in range(N)]
            otn_send(channel,messages,b'alias-table')
            yao.send(channel,circuit,yao.sender_bits(kappa,tau,pi,us,rho,ELL,P))
            index_shares.append(rho);local_coins.append((kappa,tau,pi,us,rho))
        else:
            j=secrets.randbits(ELL);ur=secrets.randbits(P)
            packed=int.from_bytes(otn_receive(channel,j,N,8,b'alias-table'),'little')
            share=yao.receive(channel,circuit,yao.receiver_bits(j,packed&((1<<(P+1))-1),packed>>(P+1),ur,ELL,P))
            index_shares.append(share);local_coins.append((j,ur))
        if (i+1)%25==0 or i+1==K:
            print('role',role,'secret proposals completed',i+1,flush=True)
    sock.close();sampled=time.perf_counter()
    proposal_metrics=dict(sent=channel.sent,received=channel.received,sends=channel.sends,receives=channel.receives)
    np.savez_compressed(folder/'proposal_private_audit.npz',coins=np.array(local_coins,dtype=np.uint64),
                        index_shares=np.array(index_shares,dtype=np.uint16))
    # Public fixed padding only; no adaptive proposal/deduplication schedule.
    padded=np.zeros(BATCH,dtype='<u2');padded[:K]=index_shares
    index_file=folder/'index_shares.bin';index_file.write_bytes(padded.tobytes())
    native_role=2 if role==0 else 1
    row_file=folder/'row_shares.bin';row_metrics_file=folder/'row_metrics.json'
    native([str(ROOT/'third_party/FABLE-smallbatch/build/bin/private_row_harness'),str(native_role),
            str(fable_port),str(BATCH),str(folder/'owner_integer_table.bin') if role==1 else '-',
            str(index_file),str(row_file),str(row_metrics_file),'4'],folder/'row_native.log')
    looked_up=time.perf_counter()
    # Each party reads only its own row output. Int32 coefficient words are
    # reduced to their18 low bits, whose XOR represents the signed18 value.
    rows=np.frombuffer(row_file.read_bytes(),dtype='<u4').reshape(BATCH,R)
    gc_input=bytearray()
    for value in (x if role==0 else np.zeros(128,dtype=np.int64)):
        gc_input.extend((int(value)&((1<<22)-1)).to_bytes(4,'little'))
    for value in (bi.ravel() if role==1 else np.zeros(8*128,dtype=np.int64)):
        gc_input.extend((int(value)&((1<<18)-1)).to_bytes(4,'little'))
    for i in range(K):
        for value in rows[i]:gc_input.extend((int(value)&((1<<18)-1)).to_bytes(4,'little'))
        gc_input.extend(int(index_shares[i]).to_bytes(4,'little'))
        for _ in range(2*19):gc_input.extend(secrets.randbits(P).to_bytes(8,'little'))
    gc_input_file=folder/'gc_input.bin';gc_input_file.write_bytes(gc_input)
    output=folder/'final_output.txt';gc_metrics_file=folder/'gc_metrics.json'
    native([str(ROOT/'toolchain/bounded-sampler/bounded_adapter_gc'),str(native_role),str(gc_port),str(K),
            str(gc_input_file),str(output),str(gc_metrics_file),'token'],folder/'gc_native.log')
    done=time.perf_counter()
    report=dict(role=role,proposals=K,padded_row_queries=BATCH,local_prepare_ms=1000*(prepared-started),
        proposal_ms=1000*(sampled-prepared),row_and_transfer_ms=1000*(looked_up-sampled),
        arithmetic_and_transfer_ms=1000*(done-looked_up),worker_total_ms=1000*(done-started),
        proposal=proposal_metrics,row=json.loads(row_metrics_file.read_bytes()),
        arithmetic=json.loads(gc_metrics_file.read_bytes()),finite_alias=table_stats,
        own_output_empty=output.stat().st_size==0)
    (folder/'worker_metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    print('role',role,'full pipeline finished',flush=True)


if __name__=='__main__':main()
