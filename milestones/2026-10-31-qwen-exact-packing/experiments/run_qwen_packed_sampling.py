"""Native-batched complete private-head chain; after-exit-only parent audit."""
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time

import numpy as np

from qwen_bounded_fixed16 import encode_parameters
from bounded_adapter_fixed16 import tanh_integer
from finite_descending_bernoulli import first_success
from run_dense_adapter_linearot import port_triple
from run_qwen_selected_exp31 import audit as audit_head
from run_dense_adapter_cdf31 import exp31
import run_fable_smallbatch_check as transport
from qwen_exactpacking_budget import reserve
from qwen_exact_row_codec import unpack18,signed18

ROOT=Path(__file__).resolve().parents[1]
K=201;N=262144


def sha(path):return transport.sha(path)


def pair(client_input,server_input,folder):
    left,right=socket.socketpair()
    sockets=[left,right];processes=[];logs=[]
    fable_port=transport.port();gc_port=port_triple()
    while fable_port in (gc_port,gc_port+50,gc_port+100):gc_port=port_triple()
    started=time.perf_counter()
    try:
        for role,sock in enumerate(sockets):
            sub=folder/f'party{role}';sub.mkdir()
            log=(sub/'worker.log').open('xb');logs.append(log)
            command=[str(ROOT/'envs/torch-cpu/bin/python'),str(ROOT/'experiments/qwen_packed_sampling_worker.py'),
                str(role),str(sock.fileno()),str(client_input if role==0 else server_input),str(sub),str(fable_port),str(gc_port)]
            processes.append(subprocess.Popen(command,pass_fds=(sock.fileno(),),stdout=log,stderr=subprocess.STDOUT,
                cwd=ROOT,start_new_session=True))
        for sock in sockets:sock.close()
        # No phase files, indices, rows, scores or coins are read here.
        while any(p.poll() is None for p in processes):
            if any(p.poll() not in (None,0) for p in processes):raise RuntimeError('party failed; retain logs')
            if time.perf_counter()-started>600:raise TimeoutError('registered full-pair timeout')
            time.sleep(.25)
        assert all(p.returncode==0 for p in processes)
    finally:
        for sock in sockets:sock.close()
        for p in processes:
            if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
        for p in processes:
            try:p.wait(5)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
        for log in logs:log.close()
    return 1000*(time.perf_counter()-started)


def audit(folder,xi,bi,ai):
    # This function is called only after both workers and their native children exit.
    parties=[folder/f'party{i}' for i in range(2)]
    metrics=[json.loads((p/'worker_metrics.json').read_bytes()) for p in parties]
    for stage in ('proposal','row','arithmetic'):
        assert metrics[0][stage]['sent']==metrics[1][stage]['received']
        assert metrics[1][stage]['sent']==metrics[0][stage]['received']
    with np.load(parties[0]/'client_alias_audit.npz') as archive:
        cut=archive['cut'];alternate=archive['alias'];mass=archive['mass']
    assert sum(map(int,mass))==N*(1<<44) and int(mass[151936:].sum())==0
    own=[]
    for p in parties:
        with np.load(p/'proposal_private_audit.npz') as archive:
            own.append({n:archive[n].copy() for n in ('columns','uniforms','index_shares')})
    indices=own[0]['index_shares']^own[1]['index_shares']
    columns=own[0]['columns']^own[1]['columns']
    uniforms=own[0]['uniforms']^own[1]['uniforms']
    for i in range(K):
        column=int(columns[i]);uniform=int(uniforms[i])
        expected=column if uniform<int(cut[column]) else int(alternate[column])
        assert indices[i]==expected and indices[i]<151936
    alias_rows=[np.frombuffer((p/'alias_row_shares.bin').read_bytes(),dtype=np.uint8).reshape(256,32) for p in parties]
    packed=(alias_rows[0]^alias_rows[1]).copy().view('<u8').reshape(256,4)
    padded_columns=np.zeros(256,dtype=np.int64);padded_columns[:K]=columns
    expected=cut[padded_columns].astype(np.uint64)|(alternate[padded_columns].astype(np.uint64)<<45)
    assert np.array_equal(packed[:,0],expected) and not packed[:,1:].any()
    for j,p in enumerate(parties):
        raw=(p/'alias_finish_input.bin').read_bytes();assert len(raw)==48*K
        out=np.frombuffer((p/'alias_index_output.bin').read_bytes(),dtype='<u4')
        assert np.array_equal(out,own[j]['index_shares'])
        for i in range(K):
            cell=raw[i*48:(i+1)*48]
            assert int.from_bytes(cell[:4],'little')==own[j]['columns'][i]
            assert cell[4:36]==alias_rows[j][i].tobytes()
            assert int.from_bytes(cell[36:44],'little')==own[j]['uniforms'][i]
            if j==0:assert int.from_bytes(cell[44:48],'little')==out[i]
            else:assert int.from_bytes(cell[44:48],'little')==0
    rows=[np.frombuffer((p/'row_shares.bin').read_bytes(),dtype=np.uint8).reshape(256,32) for p in parties]
    reconstructed=signed18((rows[0]^rows[1]).copy().view('<u4').reshape(256,8))
    expected_rows=np.vstack((ai[indices],np.repeat(ai[0][None],55,axis=0)))
    assert np.array_equal(reconstructed,expected_rows)
    for p,ar,rr in zip(parties,alias_rows,rows):
        packed=np.frombuffer((p/'alias_packed_shares.bin').read_bytes(),dtype=np.uint8).reshape(256,8)
        assert np.array_equal(ar[:,:8],packed) and not ar[:,8:].any()
        packed=np.frombuffer((p/'row_packed_shares.bin').read_bytes(),dtype=np.uint8).reshape(256,18)
        assert np.array_equal(unpack18(packed),rr.copy().view('<u4').reshape(256,8))
    # Verify actual private row/index shares were forwarded unchanged; coins
    # are generated independently within each logical party.
    encoded=[(p/'gc_input.bin').read_bytes() for p in parties];at=[0,0]
    def word(party,size):
        start=at[party];at[party]+=size
        return int.from_bytes(encoded[party][start:start+size],'little')
    def signed(v,width):return v-(1<<width) if v & (1<<(width-1)) else v
    def both(width,size):return (word(0,size)^word(1,size))&((1<<width)-1)
    xx=[signed(both(23,4),23) for _ in range(896)]
    bb=[[signed(both(18,4),18) for _ in range(896)] for _ in range(8)]
    assert xx==xi.tolist() and bb==bi.tolist()
    t=[sum(int(x)*int(b) for x,b in zip(xx,row))//65536 for row in bb]
    scores=[];accepted=[];uniforms=[]
    factors=json.loads((ROOT/'results/sci_exp31_finite_certificate.json').read_bytes())['actual_public_factors']['tables']
    row_u32=[np.frombuffer((p/'row_shares.bin').read_bytes(),dtype='<u4').reshape(256,8) for p in parties]
    for i in range(K):
        aa=[]
        for k in range(8):
            pieces=[word(j,4) for j in range(2)]
            assert all(pieces[j]==(int(row_u32[j][i,k])&((1<<18)-1)) for j in range(2))
            aa.append(signed(pieces[0]^pieces[1],18))
        shares=[word(j,4) for j in range(2)]
        assert all(shares[j]==own[j]['index_shares'][i] for j in range(2))
        index=shares[0]^shares[1];assert index==indices[i] and aa==ai[index].tolist()
        coins=[[both(44,8) for _ in range(19)] for _ in range(2)]
        z=sum(int(a)*int(v) for a,v in zip(aa,t))//65536
        delta=int(tanh_integer(np.array([z],dtype=np.int64))[0]);scores.append(delta)
        uniform=int(coins[0][0])&((1<<31)-1);uniforms.append(uniform)
        accepted.append(int(uniform<int(exp31(np.array([delta],dtype=np.int64),factors)[0])))
    assert at==list(map(len,encoded))
    audit_head([parties[1],parties[0]],xi,bi,ai[indices],indices,uniforms)
    expected=first_success(indices.tolist(),accepted,262144)
    assert (parties[1]/'final_output.txt').read_bytes()==b''
    actual=int((parties[0]/'final_output.txt').read_text());assert actual==expected
    for stage in ('alias_lookup','alias_finish'):
        assert metrics[0][stage]['sent']==metrics[1][stage]['received']
        assert metrics[1][stage]['sent']==metrics[0][stage]['received']
    stage_bytes={stage:sum(m[stage]['sent'] for m in metrics) for stage in ('proposal','row','arithmetic')}
    return dict(all_correct=True,proposals_checked=K,integer_rows_checked=256,final_token_matches=True,
        total_wire_bytes=sum(stage_bytes.values()),stage_wire_bytes=stage_bytes,parties=metrics,
        public_failure=actual==262144,
        audit_file_sha256={str(p.relative_to(ROOT)):sha(p) for folder in parties for p in folder.iterdir() if p.is_file()})


def main():
    registration=ROOT/'notes/20261031_qwen_packed_sampling_preregistration.json'
    spec=json.loads(registration.read_bytes())
    for name,digest in spec['input_sha256'].items():assert sha(ROOT/name)==digest,name
    destination=ROOT/'results/qwen_packed_sampling.json';assert not destination.exists()
    folder=ROOT/'results/qwen_packed_sampling_runs';folder.mkdir()
    with np.load(ROOT/'results/bounded_adapter_qwen_transfer_weights.npz') as weights:
        ai,bi=encode_parameters(weights['bounded_A'],weights['bounded_B'])
    with np.load(ROOT/'results/bounded_adapter_qwen_frozen_captures.npz') as captures:
        captured={k:captures[k].copy() for k in captures.files}
    results=[]
    for corpus in spec['corpora']:
        sub=folder/corpus;sub.mkdir()
        xi=np.rint(captured[corpus+'_x'].astype(np.float64)*65536).astype(np.int64)
        assert np.linalg.norm(captured[corpus+'_x'])<=52
        client_input=sub/'client_initial.npz';server_input=sub/'server_initial.npz'
        np.savez_compressed(client_input,x_integer=xi,q=captured[corpus+'_q'])
        np.savez_compressed(server_input,a_integer=ai,b_integer=bi)
        reserve("complete_rejection:"+corpus,4)
        wall_ms=pair(client_input,server_input,sub)
        row=audit(sub,xi,bi,ai);row.update(corpus=corpus,whole_pair_wall_ms=wall_ms)
        results.append(row)
        print(corpus,'full private-head chain correct, bytes',row['total_wire_bytes'],'wall_ms',wall_ms,flush=True)
    out=dict(all_passed=True,results=results,preregistration_sha256=sha(registration),
        input_sha256=spec['input_sha256'],limits=spec['limits'])
    destination.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(dict(all_passed=True,complete_chain_pairs=len(results)),indent=2))


if __name__=='__main__':main()
