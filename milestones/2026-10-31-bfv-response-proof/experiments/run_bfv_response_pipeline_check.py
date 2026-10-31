"""Six fixed public synthetic circuits, full1024-column actualSEAL path."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    reg=ROOT/'notes/20261031_bfv_response_proof_preregistration.json'
    spec=json.loads(reg.read_bytes())
    continuation=ROOT/'notes/20261031_bfv_response_proof_continuation.json'
    active=json.loads(continuation.read_bytes());start=datetime.fromisoformat(active['created_utc'])
    def remaining():
        value=active['maximum_wall_seconds']-(datetime.now(timezone.utc)-start).total_seconds()
        assert value>0,'BFVstagewallbudgetexhausted'
        return value
    for name,h in spec['source_sha256'].items():assert sha(ROOT/name)==h,name
    destination=ROOT/'results/bfv_response_pipeline_check.json';assert not destination.exists()
    folder=ROOT/'results/bfv_response_pipeline_runs';assert folder.exists() and not list(folder.iterdir())
    build=ROOT/'toolchain/bfv-response-pipeline';assert build.exists()
    archive=ROOT/'third_party/FABLE-qwen18-64/build/lib/libfable-GC.a'
    utils_object=ROOT/active['frozen_utils_object'];assert sha(utils_object)==active['frozen_utils_object_sha256']
    fmt=ROOT/'envs/fable-build/lib/libfmt.so.12.2.0'
    cmake='\n'.join(['cmake_minimum_required(VERSION 3.15)','project(bfv_response LANGUAGES CXX)',
        'set(CMAKE_CXX_STANDARD 17)','find_package(SEAL 4.1 REQUIRED)',
        f'add_executable(bfv_response "{ROOT}/experiments/bfv_response_pipeline_check.cpp")',
        f'target_link_libraries(bfv_response PRIVATE "{utils_object}" SEAL::seal "{fmt}")',''])
    (build/'CMakeLists.txt').write_text(cmake)
    commands=[[str(ROOT/'envs/fable-build/bin/cmake'),'-S',str(build),'-B',str(build/'build'),
        '-DCMAKE_CXX_COMPILER=/usr/bin/g++','-DCMAKE_BUILD_TYPE=Release',
        '-DSEAL_DIR='+str(ROOT/'third_party/FABLE/build/seal_4_1/lib/cmake/SEAL-4.1')],
        [str(ROOT/'envs/fable-build/bin/cmake'),'--build',str(build/'build'),'--parallel','2']]
    with (build/'build.log').open('w') as log:
        for cmd in commands:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=remaining())
    (build/'build_commands.json').write_text(json.dumps(commands,indent=2)+'\n')
    binary=build/'build/bfv_response'
    frozen={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'experiments/bfv_response_pipeline_check.cpp',binary,build/'CMakeLists.txt',archive,utils_object]}
    (folder/'implementation_manifest.json').write_text(json.dumps(frozen,indent=2)+'\n')
    ledger=dict(cases_reserved=0,cases_completed=0,events=[],stage_closed=False)
    def save(): (folder/'ledger.json').write_text(json.dumps(ledger,indent=2)+'\n')
    save(); rows=[]
    s=np.arange(8192,dtype=np.uint64);mask=(1<<17)-1
    for c in range(6):
        assert ledger['cases_reserved']<spec['maximum_full_column_circuits']
        fixture=folder/f'case{c}.bin';metrics=folder/f'case{c}.json'
        event=dict(case=c,status='reserved',reserved_utc=datetime.now(timezone.utc).isoformat())
        ledger['cases_reserved']+=1;ledger['events'].append(event);save()
        with (folder/f'case{c}.log').open('w') as log:
            process=subprocess.run([str(binary),str(c),str(fixture),str(metrics)],stdout=log,stderr=subprocess.STDOUT,timeout=remaining())
        event.update(exit_code=process.returncode,status='terminal',completed_utc=datetime.now(timezone.utc).isoformat());save()
        assert process.returncode==0,(c,process.returncode)
        words=np.frombuffer(fixture.read_bytes(),dtype='<u4').reshape(8192,3)
        if c==0:t=np.zeros(8192,dtype=np.uint64);value=(t*257+s*17+1)&mask
        elif c==1:t=np.full(8192,1023,dtype=np.uint64);value=np.full(8192,mask,dtype=np.uint64)
        elif c==2:t=s%1024;value=np.where((t+s)&1,mask,0)
        elif c==3:t=np.where(s%4,1024,(s*37+17)%1024);value=((t+1)*(s+3)*13)&mask
        elif c==4:t=np.where(s%2,255,512);value=np.where(s%2,1,mask)
        else:t=np.where(s%3,1024,(s*131+997)%1024);value=((t*7)^(s*31)^mask)&mask
        value=np.where(t<1024,value,0)
        assert np.array_equal(words[:,0],t) and np.array_equal(words[:,1],value) and np.array_equal(words[:,2],value)
        row=json.loads(metrics.read_bytes());assert row['all_passed'] and row['case']==c
        row['independent_Python_expected_passed']=True;rows.append(row)
        ledger['cases_completed']+=1;save()
        print(f'BFV case{c}:1024columns/8192slots exact, {row["elapsed_seconds"]:.3f}s',flush=True)
    for name,h in frozen.items():assert sha(ROOT/name)==h,name
    completed=datetime.now(timezone.utc);elapsed=(completed-start).total_seconds();assert elapsed<active['maximum_wall_seconds']
    ledger.update(stage_closed=True,completed_utc=completed.isoformat());save()
    paths=[reg,continuation]+[p for p in folder.iterdir() if p.is_file()]
    result=dict(all_passed=True,results=rows,full_column_circuits=6,total_decoded_slots_checked=6*8192,
        two_party_pairs=0,created_utc=active['created_utc'],original_created_utc=spec['created_utc'],completed_utc=completed.isoformat(),
        native_checks_completion_wall_upper_seconds=elapsed,maximum_stage_wall_seconds=active['maximum_wall_seconds'],
        implementation_sha256=frozen,files={str(p.relative_to(ROOT)):sha(p) for p in paths},
        source_sha256=spec['source_sha256'],limits=spec['limits']+' Nativecorrectness doesnotprove residualbounds orsecurity. Freshpublickeyzero/218bitconfigurationnotadoptedbyoldbackend; nocostcomparison.')
    destination.write_text(json.dumps(result,indent=2)+'\n');print('AllsixBFVpublicsyntheticcircuitscorrect',flush=True)


if __name__=='__main__':main()
