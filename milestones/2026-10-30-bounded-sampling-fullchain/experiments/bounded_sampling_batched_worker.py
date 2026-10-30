"""Batch private alias table reads using existing FABLE, then private adapter.

Same finite functionality as the sequential prototype. Wide256 table backend
is deliberately reused unchanged; this is a supporting implementation baseline.
"""
import json
import os
from pathlib import Path
import secrets
import sys
import time

import numpy as np
from finite_alias_table import build
from bounded_sampling_pipeline_worker import native,ROOT,N,K,BATCH,P,R


def main():
    if len(sys.argv)!=7:raise ValueError('role reservedfd input ownfolder fableport gcport')
    role=int(sys.argv[1]);assert role in (0,1)
    os.close(int(sys.argv[2]))  # reserved socket unused by native batch backend
    folder=Path(sys.argv[4]);fable_port=int(sys.argv[5]);gc_port=int(sys.argv[6])
    started=time.perf_counter()
    with np.load(sys.argv[3]) as own:
        if role==0:
            x=own['x_integer'].astype(np.int64)
            cut,alternate,mass,represented,table_stats=build(own['q'],P)
            assert len(cut)==N and table_stats['padded_mass']==0
            np.savez_compressed(folder/'client_alias_audit.npz',cut=cut,alias=alternate,mass=mass)
            packed=np.zeros((N,4),dtype='<u8')
            packed[:,0]=cut.astype(np.uint64)|(alternate.astype(np.uint64)<<(P+1))
            (folder/'owner_alias_table.bin').write_bytes(packed.tobytes())
        else:
            ai=own['a_integer'].astype(np.int64);bi=own['b_integer'].astype(np.int64)
            assert ai.shape==(50304,8) and bi.shape==(8,128)
            table=np.zeros((N,R),dtype='<i4');table[:50304]=ai
            (folder/'owner_integer_table.bin').write_bytes(table.tobytes())
            table_stats=None
    prepared=time.perf_counter()
    columns=np.array([secrets.randbits(16) for _ in range(K)],dtype='<u2')
    padded=np.zeros(BATCH,dtype='<u2');padded[:K]=columns
    column_file=folder/'column_shares.bin';column_file.write_bytes(padded.tobytes())
    native_role=1 if role==0 else 2  # first LUT owner is client
    alias_rows=folder/'alias_row_shares.bin';alias_metrics_file=folder/'alias_row_metrics.json'
    native([str(ROOT/'third_party/FABLE-smallbatch/build/bin/private_row_harness'),str(native_role),
        str(fable_port),'256',str(folder/'owner_alias_table.bin') if role==0 else '-',
        str(column_file),str(alias_rows),str(alias_metrics_file),'4'],folder/'alias_row_native.log')
    uniforms=np.array([secrets.randbits(44) for _ in range(K)],dtype=np.uint64)
    pads=np.array([secrets.randbits(16) if role==0 else 0 for _ in range(K)],dtype=np.uint16)
    raw_rows=alias_rows.read_bytes();finish_input=bytearray()
    for i in range(K):
        finish_input.extend(int(columns[i]).to_bytes(4,'little'))
        finish_input.extend(raw_rows[i*32:(i+1)*32])
        finish_input.extend(int(uniforms[i]).to_bytes(8,'little'))
        finish_input.extend(int(pads[i]).to_bytes(4,'little'))
    finish_file=folder/'alias_finish_input.bin';finish_file.write_bytes(finish_input)
    index_file=folder/'alias_index_output.bin';finish_metrics=folder/'alias_finish_metrics.json'
    native([str(ROOT/'toolchain/bounded-sampler/bounded_alias_batch'),str(native_role),str(gc_port),
            str(finish_file),str(index_file),str(finish_metrics)],folder/'alias_finish_native.log')
    index_shares=np.frombuffer(index_file.read_bytes(),dtype='<u2').copy();assert len(index_shares)==K
    if role==0:assert np.array_equal(index_shares,pads)
    np.savez_compressed(folder/'proposal_private_audit.npz',columns=columns,uniforms=uniforms,index_shares=index_shares)
    sampled=time.perf_counter()
    alias_lookup=json.loads(alias_metrics_file.read_bytes());alias_finish=json.loads(finish_metrics.read_bytes())
    proposal_metrics={key:alias_lookup[key]+alias_finish[key] for key in ('sent','received')}
    padded=np.zeros(BATCH,dtype='<u2');padded[:K]=index_shares
    index_file=folder/'index_shares.bin';index_file.write_bytes(padded.tobytes())
    native_role=2 if role==0 else 1
    rows_file=folder/'row_shares.bin';row_metrics=folder/'row_metrics.json'
    # Reuse public ports only after prior native processes have returned. NetIO
    # uses SO_REUSEADDR, allowing the owner-role reversal on loopback.
    native([str(ROOT/'third_party/FABLE-smallbatch/build/bin/private_row_harness'),str(native_role),
        str(fable_port),'256',str(folder/'owner_integer_table.bin') if role==1 else '-',
        str(index_file),str(rows_file),str(row_metrics),'4'],folder/'row_native.log')
    looked_up=time.perf_counter()
    rows=np.frombuffer(rows_file.read_bytes(),dtype='<u4').reshape(BATCH,R)
    gc_input=bytearray()
    for value in (x if role==0 else np.zeros(128,dtype=np.int64)):
        gc_input.extend((int(value)&((1<<22)-1)).to_bytes(4,'little'))
    for value in (bi.ravel() if role==1 else np.zeros(8*128,dtype=np.int64)):
        gc_input.extend((int(value)&((1<<18)-1)).to_bytes(4,'little'))
    for i in range(K):
        for value in rows[i]:gc_input.extend((int(value)&((1<<18)-1)).to_bytes(4,'little'))
        gc_input.extend(int(index_shares[i]).to_bytes(4,'little'))
        for _ in range(38):gc_input.extend(secrets.randbits(P).to_bytes(8,'little'))
    gc_file=folder/'gc_input.bin';gc_file.write_bytes(gc_input)
    output=folder/'final_output.txt';arithmetic_file=folder/'gc_metrics.json'
    native([str(ROOT/'toolchain/bounded-sampler/bounded_adapter_gc'),str(native_role),str(gc_port),str(K),
            str(gc_file),str(output),str(arithmetic_file),'token'],folder/'gc_native.log')
    done=time.perf_counter()
    report=dict(role=role,proposals=K,padded_row_queries=BATCH,
        local_prepare_ms=1000*(prepared-started),proposal_ms=1000*(sampled-prepared),
        row_and_transfer_ms=1000*(looked_up-sampled),arithmetic_and_transfer_ms=1000*(done-looked_up),
        worker_total_ms=1000*(done-started),proposal=proposal_metrics,
        alias_lookup=alias_lookup,alias_finish=alias_finish,
        row=json.loads(row_metrics.read_bytes()),arithmetic=json.loads(arithmetic_file.read_bytes()),
        finite_alias=table_stats,own_output_empty=output.stat().st_size==0)
    (folder/'worker_metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    print('role',role,'batched full pipeline finished',flush=True)


if __name__=='__main__':main()
