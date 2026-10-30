// Fixed-shape acceptance and first-success release for bounded finite scores.
// Uses existing SCI semi-honest GC; no new garbling or Bernoulli primitive.
// Input files contain only one party's XOR shares. No score, proposal or
// acceptance flag is revealed. Only BOB receives the final token/failure code.
#include "GC/emp-sh2pc.h"
#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

using namespace sci;
using Clock = std::chrono::steady_clock;
constexpr int P=44, F=20, L=19, PER=22+16+2*L*P;

static uint64_t read_word(std::ifstream& f, int bytes) {
    uint64_t out=0;
    for(int j=0;j<bytes;j++) {
        int c=f.get();
        if(c==EOF) throw std::runtime_error("short party input");
        out|=uint64_t(uint8_t(c))<<(8*j);
    }
    return out;
}

static Integer shared_word(const std::vector<Bit>& left,
                           const std::vector<Bit>& right, size_t& at, int width) {
    Integer out(width,0,PUBLIC);
    for(int j=0;j<width;j++,at++) out[j]=left[at]^right[at];
    return out;
}

static double ms(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double,std::milli>(b-a).count();
}

int main(int argc,char** argv) {
    try {
        if(argc!=7) throw std::runtime_error("role port count partyinput output metrics");
        int role=std::stoi(argv[1]),port=std::stoi(argv[2]),count=std::stoi(argv[3]);
        if((role!=ALICE && role!=BOB) || count<1 || count>201)
            throw std::runtime_error("bad public shape");
        std::ifstream input(argv[4],std::ios::binary);
        if(!input) throw std::runtime_error("no party input");
        size_t n=size_t(count)*PER,at=0;
        std::unique_ptr<bool[]> local(new bool[n]);
        auto append=[&](uint64_t word,int width) {
            if(word>>width) throw std::runtime_error("bad share word width");
            for(int j=0;j<width;j++) local[at++]=(word>>j)&1;
        };
        for(int i=0;i<count;i++) {
            append(read_word(input,4),22);
            append(read_word(input,4),16);
            for(int j=0;j<2*L;j++) append(read_word(input,8),P);
        }
        if(at!=n || input.get()!=EOF) throw std::runtime_error("bad party input size");
        NetIO io(role==ALICE?nullptr:"127.0.0.1",port,true);
        auto start=Clock::now();
        setup_semi_honest(&io,role);
        auto setup=Clock::now();
        uint64_t setup_sent=io.counter,setup_recv=io.received_counter;
        static_assert(sizeof(Bit)==sizeof(block128));
        std::vector<Bit> left(n),right(n);
        prot_exec->feed(reinterpret_cast<block128*>(left.data()),ALICE,local.get(),n);
        prot_exec->feed(reinterpret_cast<block128*>(right.data()),BOB,local.get(),n);
        io.flush();
        auto fed=Clock::now();
        uint64_t input_sent=io.counter,input_recv=io.received_counter;
        auto before=circ_exec->num_and();
        Integer result(32,65536,PUBLIC);
        Bit found(false,PUBLIC);
        at=0;
        for(int i=0;i<count;i++) {
            auto delta=shared_word(left,right,at,22);
            delta.resize(P+2,true);
            auto index=shared_word(left,right,at,16);
            index.resize(32,false);
            auto threshold=(Integer(P+2,int64_t(1)<<F,PUBLIC)-delta)<<(P-F-1);
            Bit accepted(true,PUBLIC);
            for(int chain=0;chain<2;chain++) {
                Integer previous=threshold;
                Bit active(true,PUBLIC),parity(false,PUBLIC);
                for(int j=0;j<L;j++) {
                    auto uniform=shared_word(left,right,at,P);
                    uniform.resize(P+2,false);
                    active=active & (uniform<previous);
                    parity=parity ^ active;
                    previous=uniform;
                }
                accepted=accepted & ((!active)&(!parity));
            }
            auto take=accepted & (!found);
            result=result.select(take,index);
            found=found | accepted;
        }
        if(at!=n) throw std::runtime_error("internal wire count mismatch");
        io.flush();
        auto computed=Clock::now();
        uint64_t compute_sent=io.counter,compute_recv=io.received_counter;
        auto hash_counter=circ_exec->num_and()-before;
        auto table_bytes=role==ALICE?compute_sent-input_sent:compute_recv-input_recv;
        if(table_bytes%32) throw std::runtime_error("nonintegral halfgate table size");
        auto ands=table_bytes/32;
        uint32_t released=result.reveal<uint32_t>(BOB);
        io.flush();
        auto done=Clock::now();
        std::ofstream out(argv[5]);
        if(role==BOB) out<<released<<"\n";
        out.close();
        std::ofstream metrics(argv[6]);
        metrics<<"{\"role\":"<<role<<",\"count\":"<<count
          <<",\"precision\":44,\"draws_per_factory\":19,\"and_gates\":"<<ands
          <<",\"upstream_hash_key_counter_delta\":"<<hash_counter
          <<",\"setup_ms\":"<<ms(start,setup)<<",\"input_ms\":"<<ms(setup,fed)
          <<",\"compute_ms\":"<<ms(fed,computed)<<",\"release_ms\":"<<ms(computed,done)
          <<",\"protocol_ms\":"<<ms(start,done)
          <<",\"sent\":"<<io.counter<<",\"received\":"<<io.received_counter
          <<",\"setup_sent\":"<<setup_sent<<",\"setup_received\":"<<setup_recv
          <<",\"input_sent\":"<<input_sent-setup_sent<<",\"input_received\":"<<input_recv-setup_recv
          <<",\"compute_sent\":"<<compute_sent-input_sent<<",\"compute_received\":"<<compute_recv-input_recv
          <<",\"release_sent\":"<<io.counter-compute_sent<<",\"release_received\":"<<io.received_counter-compute_recv
          <<",\"io_direction_changes\":"<<io.num_rounds<<"}\n";
        std::cout<<"ACCEPT_SELECT_FINISHED role="<<role<<std::endl;
        return 0;
    } catch(const std::exception& e) {
        std::cerr<<"accept_select error: "<<e.what()<<std::endl;
        return 1;
    }
}
