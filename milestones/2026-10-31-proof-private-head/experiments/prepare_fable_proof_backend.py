"""Make isolated registered FABLE variants; never mutate measured baselines."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT=Path(__file__).resolve().parents[1]


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def replace(path,old,new,count=1):
    text=path.read_text();assert text.count(old)==count,(path,old[:70],text.count(old))
    path.write_text(text.replace(old,new))


def isolate_SCI(new,old):
    target=new/'extern/EzPC'
    if target.is_symlink():target.unlink()
    target.mkdir(exist_ok=True)
    source=(old/'extern/EzPC/SCI').resolve();private=target/'SCI';private.mkdir()
    shutil.copy2(source/'CMakeLists.txt',private/'CMakeLists.txt')
    shutil.copytree(source/'cmake',private/'cmake')
    for name in ('src','extern','tests','networks'):
        (private/name).symlink_to(source/name,target_is_directory=True)


def make(width):
    old=ROOT/f'third_party/FABLE-qwen18-{width}'
    new=ROOT/f'third_party/FABLE-qwen-proof-{width}'
    assert not new.exists();new.mkdir()
    shutil.copy2(old/'CMakeLists.txt',new/'CMakeLists.txt')
    for name in ('src','test'):
        shutil.copytree(old/name,new/name,symlinks=True)
    (new/'extern').mkdir()
    shutil.copy2(old/'extern/CMakeLists.txt',new/'extern/CMakeLists.txt')
    shutil.copytree(old/'extern/BatchPIR',new/'extern/BatchPIR',symlinks=True,
                    ignore=shutil.ignore_patterns('.git','build','build_*'))
    isolate_SCI(new,old)
    header=new/'extern/BatchPIR/header';source=new/'extern/BatchPIR/src'
    shutil.copy2(ROOT/'experiments/private_lut_placement.h',header/'private_lut_placement.h')
    shutil.copy2(ROOT/'experiments/fable_dual_domain_aes.h',new/'src/GC/fable_dual_domain_aes.h')
    replace(header/'utils.h','#include "LowMC.h"','#include "LowMC.h"\n#include "private_lut_placement.h"')
    p=header/'utils.h';text=p.read_text();a=text.index('    inline std::size_t hash_mod(');b=text.index('    template< size_t size = blocksize>',a)
    text=text[:a]+'''    inline std::vector<size_t> proof_candidates(const string& tags, size_t count, size_t slots, size_t domain) {
        check(tags.size()==256 && count==3 && domain<2,"fixeddual-domaintriple profile");
        __uint128_t value=0;
        for (size_t j=0;j<128;++j) {
            const char bit=tags[domain*128+j];
            check(bit=='0'||bit=='1',"tagencoding");
            value=(value<<1)|(__uint128_t)(bit-'0');
        }
        auto triple=::private_lut::balanced_triple(value,slots);
        return {triple[0],triple[1],triple[2]};
    }
    inline std::vector<size_t> get_candidate_buckets(string tags,size_t count,size_t slots) {
        return proof_candidates(tags,count,slots,0);
    }
    inline std::vector<size_t> get_candidate_positions(string tags,size_t count,size_t slots) {
        return proof_candidates(tags,count,slots,1);
    }

'''+text[b:]
    for line in text.splitlines(keepends=True):
        if 'bool cuckoo_insert(' in line:text=text.replace(line,'')
    text=text.replace('PlaintextModBitss = 20;\n                    CoeffMods = vector<int>{45, 45, 50};\n                    noise_bits = 69;',
                      'PlaintextModBitss = 18;\n                    CoeffMods = vector<int>{46, 40, 40, 46, 46};\n                    noise_bits = 149;')
    p.write_text(text)
    p=source/'utils.cpp';text=p.read_text();a=text.index('    bool cuckoo_insert(');b=text.index('    // sample r <-',a);p.write_text(text[:a]+text[b:])
    replace(header/'database_constants.h','    constexpr int MaxAttempts = 500;\n','')
    p=source/'batchpirparams.cpp'
    replace(p,'    std::string selection =','    utils::check(batch_size==256 && db_size==262399 && type==PIRANA && hash_type==AES && LUT_INPUT_SIZE==18 && (LUT_OUTPUT_SIZE==64 || LUT_OUTPUT_SIZE==144), "fixedproofbackendprofile");\n    std::string selection =')
    replace(p,'return ceil(CuckooFactorBucket * MaxBucketSize);','return 4096; // registered complete-matching position capacity')
    replace(p,'std::cout << std::left << std::setw(20) << "| MaxAttempts: " << MaxAttempts << std::endl;',
              'std::cout << "| CompleteMatching: true" << std::endl;')
    p=source/'batchpirclient.cpp';text=p.read_text();a=text.index('    std::unordered_map<uint64_t, std::vector<uint64_t>> key_to_buckets');b=text.index('    is_cuckoo_generated_ = true;',a)
    text=text[:a]+'''    ::private_lut::Matching matching(total_buckets);
    vector<vector<size_t>> key_to_position(batch_size);
    for (size_t i=0;i<batch_size;++i) {
        auto buckets=utils::get_candidate_buckets(batch[i],w,total_buckets);
        key_to_position[i]=utils::get_candidate_positions(batch[i],w,bucket_size);
        utils::check(matching.insert({buckets[0],buckets[1],buckets[2]}),"completequerymatchingfailed");
    }
    utils::check(matching.valid(),"querymatchinginvalid");
    cuckoo_map.clear();inv_cuckoo_map.resize(batch_size);
    bucket_to_position.assign(total_buckets,vector<uint64_t>(w,batchpir_params_->get_default_value()));
    for (size_t key=0;key<batch_size;++key) {
        auto bucket=matching.assignment()[key];
        cuckoo_map[bucket]=key;inv_cuckoo_map[key]=bucket;
        bucket_to_position[bucket]=key_to_position[key];
    }

'''+text[b:]
    a=text.index('        auto glk = keygen_->create_galois_keys();');b=text.index('        relin_keys_.load(',a)
    text=text[:a]+'''        seal::PublicKey pk;keygen_->create_public_key(pk);
        auto rlk=keygen_->create_relin_keys();
        glk_buffer.resize(pk.save_size());rlk_buffer.resize(rlk.save_size());
        auto pk_size=pk.save(glk_buffer.data(),glk_buffer.size());
        auto rlk_size=rlk.save(rlk_buffer.data(),rlk_buffer.size());
        glk_buffer.resize(pk_size);rlk_buffer.resize(rlk_size);
'''+text[b:]
    text=text.replace('public_keys = std::make_pair(vector<seal_byte>{}, rlk_buffer);','public_keys = std::make_pair(glk_buffer, rlk_buffer); // first buffer is BFV PublicKey, no GaloisKeys')
    p.write_text(text)
    p=header/'batchpirserver.h';replace(p,'    void hash_encode();','    void complete_public_id_domain();\n    void hash_encode();')
    replace(p,'    std::map<uint32_t, seal::RelinKeys> client_keys_;','    std::map<uint32_t, seal::RelinKeys> client_keys_;\n    std::map<uint32_t, seal::PublicKey> client_public_keys_;')
    p=source/'batchpirserver.cpp';text=p.read_text();text=text.replace('    is_db_populated = true;','    complete_public_id_domain();')
    # Four population paths all start from the original full public table.
    text=text.replace('    size_t db_entries = batchpir_params_->get_db_size();\n    int num_threads = batchpir_params_->get_num_threads();',
                      '    size_t db_entries = 262144;\n    int num_threads = batchpir_params_->get_num_threads();',1)
    a=text.index('void BatchPIRServer::initialize_masks()')
    text=text[:a]+'''void BatchPIRServer::complete_public_id_domain() {
    check(input_keys.size()==262144 && rawdb_.size()==262144,"canonicalfulltable required");
    for (size_t i=0;i<262144;++i) check(input_keys[i]==i,"canonicalrowIDs required");
    for (size_t i=1;i<256;++i) {input_keys.push_back(262144+i);rawdb_.push_back(rawdatablock(0));}
    check(input_keys.size()==batchpir_params_->get_db_size(),"virtualIDdomain size");
    is_db_populated=true;
}

'''+text[a:]
    a=text.index('    uint64_t high_prefix;',text.index('void BatchPIRServer::aes_prepare'));b=text.index('\n}\n',a)
    text=text[:a]+'''    check(oprf_prefix.none(),"fixedpublicAESprefix required");
    #pragma omp parallel for if(parallel) num_threads(num_threads)
    for (uint64_t i=0;i<db_entries;++i) {
        string tags;
        for (int domain=0;domain<2;++domain) {
            alignas(16) uint64_t data[2]{input_keys[i],uint64_t(domain)<<63};
            auto c=aes_oprf->ecbEncBlock(oc::block(_mm_load_si128((__m128i*)data))).get<uint64_t>();
            tags+=std::bitset<64>(c[1]).to_string()+std::bitset<64>(c[0]).to_string();
        }
        candidate_buckets_array[i]=get_candidate_buckets(tags,w,total_buckets);
        candidate_positions_array[i]=get_candidate_positions(tags,w,bucket_size);
    }
'''+text[b:]
    a=text.index('    srand(time(nullptr));',text.index('void BatchPIRServer::hash_encode'));text=text[:a]+text[a:].replace('    srand(time(nullptr)); // Used for local cuckoo hashing, no need to be cryptographically secure. \n','',1)
    oldloop='''        for (auto& i: insert_buffer[bucket_idx]) {
            cuckoo_insert(i, 0, candidate_positions_array, position_to_key[bucket_idx]);
        }'''
    assert text.count(oldloop)==1
    text=text.replace(oldloop,'''        ::private_lut::Matching matching(bucket_size);
        for (auto id:insert_buffer[bucket_idx]) {
            auto& pos=candidate_positions_array[id];
            check(matching.insert({pos[0],pos[1],pos[2]}),"completeservermatchingfailed");
        }
        check(matching.valid(),"servermatchinginvalid");
        for (size_t k=0;k<insert_buffer[bucket_idx].size();++k)
            position_to_key[bucket_idx][matching.assignment()[k]]=insert_buffer[bucket_idx][k];''')
    text=text.replace('        auto [glk_buffer, rlk_buffer] = keys;\n        seal::RelinKeys rlk;',
                      '        auto [glk_buffer, rlk_buffer] = keys;\n        seal::PublicKey pk;pk.load(*context_,glk_buffer.data(),glk_buffer.size());\n        client_public_keys_[client_id]=pk;\n        seal::RelinKeys rlk;',1)
    needle='                NoiseFloodInplace(response[hash_idx][slot_idx], *context_, batchpir_params_->noise_bits);'
    assert text.count(needle)==1
    text=text.replace(needle,'''                seal::Encryptor rerandomizer(*context_,client_public_keys_.at(client_id));
                seal::Ciphertext zero;
                rerandomizer.encrypt_zero(response[hash_idx][slot_idx].parms_id(),zero);
                evaluator_->add_inplace(response[hash_idx][slot_idx],zero);
'''+needle)
    needle='            vector<PIRResponseList> masked_value(num_columns_per_entry, PIRResponseList(active_task_groups));'
    assert text.count(needle)==1
    text=text.replace(needle,'''            // Every selected record, including an empty cell, is fully masked.
            // Sentinel524287 is outside the covered19-bit query-ID domain.
            vector<vector<uint64_t>> masked_empty(num_columns_per_entry,vector<uint64_t>(num_buckets));
            for (size_t bucket_idx=0;bucket_idx<num_buckets;++bucket_idx) {
                auto value=(rawinputblock((1ULL<<InputLength)-1)^index_masks[hash_idx][bucket_idx]).to_string()
                    +entry_masks[hash_idx][bucket_idx].to_string();
                for (size_t slot_idx=0;slot_idx<num_columns_per_entry;++slot_idx) {
                    size_t start=slot_idx*size_of_coeff;
                    size_t end=std::min((slot_idx+1)*size_of_coeff,(size_t)datablock_size);
                    masked_empty[slot_idx][bucket_idx]=std::stoull(value.substr(start,end-start),0,2);
                }
            }
'''+needle)
    needle='                                    plain_col[slot_idx].push_back(0);'
    assert text.count(needle)==1
    text=text.replace(needle,'                                    plain_col[slot_idx].push_back(masked_empty[slot_idx][bucket_idx]);')
    p.write_text(text)
    p=new/'src/GC/lookup.cpp';text=p.read_text();text=text.replace('#include "lookup.h"','#include "lookup.h"\n#include "fable_dual_domain_aes.h"',1)
    needle='new BatchPirParams(batch_size, lut.size(), parallel, num_threads, (BatchPirType)type, (HashType)hash_type)'
    assert text.count(needle)==2
    text=text.replace(needle,'new BatchPirParams(batch_size, db_size+batch_size-1, parallel, num_threads, (BatchPirType)type, (HashType)hash_type)')
    needle='new BatchPirParams(batch_size, db_size, parallel, num_threads, type, hash_type)'
    assert text.count(needle)==1
    text=text.replace(needle,'new BatchPirParams(batch_size, db_size+batch_size-1, parallel, num_threads, type, hash_type)')
    text=text.replace('oc::block aes_key;','oc::block aes_key=oc::ZeroBlock;')
    text=text.replace('sci::AES* aes_ciphers_2PC;','sci::AES* aes_ciphers_2PC=nullptr;')
    # Replace only the actual unfused AES branch, keeping the public API narrow.
    a=text.index('\t\tvector<Integer> m(batch_size);',text.index('IntegerArray fable_lookup('));b=text.index('\n\t}\n\tend_record(io_gc, "OPRF Evaluation"',a)
    text=text[:a]+'        batch=proof_dual_domain_aes(secret_queries,aes_key);'+text[b:]
    p.write_text(text)
    replace(new/'CMakeLists.txt','experiments/fable_private_row_qwen_packed.cpp','experiments/fable_private_row_proof.cpp')
    with (new/'CMakeLists.txt').open('a') as f:
        f.write('\nadd_executable(dual_domain_aes_control ${PROJECT_SOURCE_DIR}/../../experiments/fable_dual_domain_aes_control.cpp)\ntarget_link_libraries(dual_domain_aes_control fable-GC)\ntarget_compile_definitions(dual_domain_aes_control PRIVATE LUT_INPUT_SIZE=18 LUT_OUTPUT_SIZE=${LUT_OUTPUT_SIZE} LUT_MAX_LOG_SIZE=18)\n')
    return old,new


def main():
    folder=ROOT/'sources/fable_proof_backend_build';folder.mkdir(exist_ok=True)
    assert not list(folder.iterdir())
    manifest={}
    for width in (64,144):
        old,new=make(width)
        changed={}
        for p in new.rglob('*'):
            if not p.is_file() or p.is_symlink() or 'extern/EzPC' in str(p):continue
            rel=p.relative_to(new);base=old/rel
            if not base.exists() or sha(base)!=sha(p):changed[str(rel)]=sha(p)
        # Persist a reviewable patch without traversing shared dependency trees.
        patches=[]
        for name in sorted(changed):
            import difflib
            base=old/name;p=new/name
            patches.extend(difflib.unified_diff(base.read_text().splitlines(True) if base.exists() else [],
                p.read_text().splitlines(True),fromfile='a/'+name,tofile='b/'+name))
        patch=folder/f'proof{width}_over_packed.patch';patch.write_text(''.join(patches))
        manifest[str(width)]=dict(base=str(old.relative_to(ROOT)),variant=str(new.relative_to(ROOT)),
            changed_source_sha256=changed,patch_sha256=sha(patch))
    (folder/'source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:len(v['changed_source_sha256']) for k,v in manifest.items()}))


if __name__=='__main__':main()
