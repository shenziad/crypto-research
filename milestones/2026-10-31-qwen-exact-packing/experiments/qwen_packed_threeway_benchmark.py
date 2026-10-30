"""Four alternating pairs of complete samplers for the identical E31 target."""
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import statistics
import subprocess
import time

import numpy as np
from bounded_adapter_fixed16 import tanh_integer,SCALE
from finite_alias_table import build
from run_dense_adapter_linearot import input_bytes,port_triple,reconstruct,sha
from run_dense_adapter_cdf31 import exp31
from run_private_weighted_cdf95_check import audit as cdf_audit
import run_qwen_sampling_exp31 as wide
import run_qwen_packed_sampling as packed
from qwen_exactpacking_budget import reserve

ROOT=Path(__file__).resolve().parents[1]


def execute(variant,folder,common):
    processes=[];logs=[];sockets=[]
    gcport=port_triple();fableport=wide.transport.port()
    while fableport in (gcport,gcport+50,gcport+100):fableport=wide.transport.port()
    if variant!='dense':sockets=list(socket.socketpair())
    remaining=reserve("balanced:"+folder.name,1 if variant=="dense" else 4)
    started=time.perf_counter()
    try:
        for i in range(2):
            role=i if variant!='dense' else i+1
            sub=folder/f'party{role}';sub.mkdir();log=(sub/'worker.log').open('xb');logs.append(log)
            if variant!='dense':
                cmd=[str(ROOT/'envs/torch-cpu/bin/python'),str(ROOT/'experiments'/('qwen_packed_sampling_worker.py' if variant=='packed' else 'qwen_sampling_exp31_worker.py')),str(i),
                    str(sockets[i].fileno()),str(common/('client.npz' if i==0 else 'server.npz')),str(sub),str(fableport),str(gcport)]
                fds=(sockets[i].fileno(),)
            else:
                cmd=[str(ROOT/'toolchain/bounded-sampler/qwen_dense_cdf31'),str(i+1),str(gcport),'151936',
                    str(common/f'dense_party{i+1}.bin'),str(sub),'1'];fds=()
            processes.append(subprocess.Popen(cmd,cwd=ROOT,pass_fds=fds,stdout=log,stderr=subprocess.STDOUT,start_new_session=True))
        for s in sockets:s.close()
        while any(p.poll() is None for p in processes):
            if any(p.poll() not in (0,None) for p in processes):raise RuntimeError('full sampler failed; logs retained')
            if time.perf_counter()-started>min(600,remaining):raise TimeoutError('registered pair timeout')
            time.sleep(.1)
        assert all(p.returncode==0 for p in processes)
    finally:
        for s in sockets:s.close()
        for p in processes:
            if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
        for p in processes:
            try:p.wait(5)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
        for log in logs:log.close()
    return 1000*(time.perf_counter()-started)


def main():
    reg=ROOT/'notes/20261031_qwen_packed_threeway_benchmark_preregistration.json';spec=json.loads(reg.read_bytes())
    for name,digest in spec['input_sha256'].items():assert sha(ROOT/name)==digest,name
    destination=ROOT/'results/qwen_packed_threeway_benchmark.json';assert not destination.exists()
    folder=ROOT/'results/qwen_packed_threeway_benchmark_runs';folder.mkdir();common=folder/'common';common.mkdir()
    previous=ROOT/'results/qwen_sampling_exp31_runs/ptb_test'
    for source,name in [('client_initial.npz','client.npz'),('server_initial.npz','server.npz')]:
        (common/name).write_bytes((previous/source).read_bytes())
    with np.load(common/'client.npz') as archive:x=archive['x_integer'];q=archive['q']
    with np.load(common/'server.npz') as archive:a=archive['a_integer'];b=archive['b_integer']
    mass=build(q,44)[2][:151936].astype(np.uint64);assert sum(map(int,mass))==1<<62
    raw_t=x@b.T;t=raw_t//SCALE;raw_z=a@t;delta=tanh_integer(raw_z//SCALE)
    factors=json.loads((ROOT/'results/sci_exp31_finite_certificate.json').read_bytes())['actual_public_factors']['tables']
    e=exp31(delta,factors)
    for role in (1,2):
        qm=mass if role==2 else np.zeros_like(mass)
        (common/f'dense_party{role}.bin').write_bytes(input_bytes(x,b,a,role)+qm.astype('<u8').tobytes())
    records=[]
    for repetition,order in enumerate(spec['orders']):
        for variant in order:
            sub=folder/f'{repetition}_{variant}';sub.mkdir();wall=execute(variant,sub,common)
            if variant!='dense':
                row=(packed if variant=='packed' else wide).audit(sub,x,b,a)
                participant=max(p['worker_total_ms'] for p in row['parties'])
            else:
                paths=[sub/f'party{i}' for i in (1,2)]
                for name,width,value in [('raw_t_add41.bin',41,raw_t),('t_add26.bin',26,t),('raw_z_add44.bin',44,raw_z),
                    ('delta_add44.bin',44,delta),('exp_add33.bin',33,e)]:
                    assert np.array_equal(reconstruct(paths,name,width),value),name
                row=cdf_audit(sub,mass.tolist(),e.tolist())
                party=[json.loads((p/'metrics.json').read_bytes()) for p in paths]
                for channel in range(3):
                    assert party[0]['end_counters'][channel]==party[1]['end_counters'][channel+3]
                    assert party[1]['end_counters'][channel]==party[0]['end_counters'][channel+3]
                row.update(parties=party,total_wire_bytes=sum(sum(p['end_counters'][:3]) for p in party),
                    files={str(p.relative_to(ROOT)):sha(p) for folder in paths for p in folder.iterdir() if p.is_file()})
                participant=max(p['protocol_ms'] for p in party)
            row.update(variant=variant,repetition=repetition,whole_pair_wall_ms=wall,participant_elapsed_ms=participant)
            records.append(row)
            print(repetition,variant,'completecorrect, bytes',row['total_wire_bytes'],'wallms',wall,flush=True)
    summary={variant:dict(observations=4,
        median_total_wire_bytes=statistics.median(r['total_wire_bytes'] for r in records if r['variant']==variant),
        median_whole_pair_wall_ms=statistics.median(r['whole_pair_wall_ms'] for r in records if r['variant']==variant),
        median_participant_elapsed_ms=statistics.median(r['participant_elapsed_ms'] for r in records if r['variant']==variant))
        for variant in ('dense','wide','packed')}
    out=dict(all_passed=True,summary=summary,results=records,
        target_q_mass_sha256=hashlib.sha256(mass.astype('<u8').tobytes()).hexdigest(),
        target_E31_vector_sha256=hashlib.sha256(e.astype('<u8').tobytes()).hexdigest(),
        preregistration_sha256=sha(reg),input_sha256=spec['input_sha256'],limits=spec['limits'])
    destination.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
