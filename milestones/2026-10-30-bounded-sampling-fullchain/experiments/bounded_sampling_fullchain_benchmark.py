"""Alternating matched full-chain comparison of two existing backend choices."""
import json
import os
from pathlib import Path
import signal
import socket
import statistics
import subprocess
import time

import numpy as np

import run_bounded_sampling_pipeline as sequential
import run_bounded_sampling_batched as batched

ROOT=Path(__file__).resolve().parents[1]


def run(worker,client,server,folder):
    left,right=socket.socketpair();sockets=[left,right];children=[];logs=[]
    ports=[sequential.transport.port(),sequential.transport.port()]
    while ports[0]==ports[1]:ports[1]=sequential.transport.port()
    started=time.perf_counter()
    try:
        for role,sock in enumerate(sockets):
            sub=folder/f'party{role}';sub.mkdir()
            log=(sub/'worker.log').open('xb');logs.append(log)
            command=[str(ROOT/'envs/torch-cpu/bin/python'),str(ROOT/'experiments'/worker),str(role),
                str(sock.fileno()),str(client if role==0 else server),str(sub),*map(str,ports)]
            children.append(subprocess.Popen(command,cwd=ROOT,pass_fds=(sock.fileno(),),
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True))
        for sock in sockets:sock.close()
        while any(p.poll() is None for p in children):
            if any(p.poll() not in (0,None) for p in children):raise RuntimeError('pair failure; logs retained')
            if time.perf_counter()-started>600:raise TimeoutError('pair timeout')
            time.sleep(.1)
        assert all(p.returncode==0 for p in children)
    finally:
        for sock in sockets:sock.close()
        for p in children:
            if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
        for p in children:
            try:p.wait(5)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
        for log in logs:log.close()
    return 1000*(time.perf_counter()-started)


def main():
    registration=ROOT/'notes/20261030_bounded_sampling_fullchain_benchmark_preregistration.json'
    spec=json.loads(registration.read_bytes())
    for name,digest in spec['input_sha256'].items():assert sequential.sha(ROOT/name)==digest,name
    destination=ROOT/'results/bounded_sampling_fullchain_benchmark.json';assert not destination.exists()
    folder=ROOT/'results/bounded_sampling_fullchain_benchmark_runs';folder.mkdir()
    initial=ROOT/'results/bounded_sampling_pipeline_runs/ptb'
    client=initial/'client_initial.npz';server=initial/'server_initial.npz'
    with np.load(client) as source:xi=source['x_integer']
    with np.load(server) as source:ai=source['a_integer'];bi=source['b_integer']
    records=[]
    for repetition,order in enumerate(spec['orders']):
        for variant in order:
            sub=folder/f'{repetition}_{variant}';sub.mkdir()
            module=sequential if variant=='sequential' else batched
            worker='bounded_sampling_pipeline_worker.py' if variant=='sequential' else 'bounded_sampling_batched_worker.py'
            wall=run(worker,client,server,sub)
            row=module.audit(sub,xi,bi,ai)
            row.update(repetition=repetition,variant=variant,whole_pair_wall_ms=wall,
                max_worker_ms=max(p['worker_total_ms'] for p in row['parties']))
            records.append(row)
            print(repetition,variant,'allcorrect',row['total_wire_bytes'],'wallms',wall,
                  'maxworkerms',row['max_worker_ms'],flush=True)
    summary={v:dict(observations=4,
        median_total_wire_bytes=statistics.median(r['total_wire_bytes'] for r in records if r['variant']==v),
        median_max_worker_ms=statistics.median(r['max_worker_ms'] for r in records if r['variant']==v),
        median_whole_pair_wall_ms=statistics.median(r['whole_pair_wall_ms'] for r in records if r['variant']==v))
        for v in ('sequential','batched')}
    out=dict(all_passed=True,results=records,summary=summary,
        preregistration_sha256=sequential.sha(registration),input_sha256=spec['input_sha256'],limits=spec['limits'])
    destination.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
