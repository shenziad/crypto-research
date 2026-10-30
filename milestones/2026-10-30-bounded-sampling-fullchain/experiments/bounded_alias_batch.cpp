// Finish batched secret alias lookup: secret column/threshold/alternate/U,
// output fresh XOR index shares. Existing SCI GC, no new primitive.
#include "GC/emp-sh2pc.h"
#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>
using namespace sci;
using Clock=std::chrono::steady_clock;
constexpr int K=201,PER=16+45+16+44+16;

static uint64_t read_word(std::ifstream& f,int n) {
    uint64_t v=0;for(int j=0;j<n;j++) {
        int c=f.get();if(c==EOF)throw std::runtime_error("short input");
        v|=uint64_t(uint8_t(c))<<(8*j);
    }return v;
}
static Integer shared(const std::vector<Bit>& a,const std::vector<Bit>& b,size_t& at,int width) {
    Integer out(width,0,PUBLIC);for(int j=0;j<width;j++,at++)out[j]=a[at]^b[at];return out;
}
static double ms(Clock::time_point a,Clock::time_point b) {
    return std::chrono::duration<double,std::milli>(b-a).count();
}
int main(int argc,char** argv) {
    try {
        if(argc!=6)throw std::runtime_error("role port input output metrics");
        int role=std::stoi(argv[1]),port=std::stoi(argv[2]);
        if(role!=ALICE && role!=BOB)throw std::runtime_error("bad role");
        std::ifstream f(argv[3],std::ios::binary);if(!f)throw std::runtime_error("no input");
        size_t at=0,n=K*PER;std::unique_ptr<bool[]> bits(new bool[n]);
        std::vector<uint16_t> pads;
        auto append=[&](uint64_t v,int width) {
            if(v>>width)throw std::runtime_error("bad word");
            for(int j=0;j<width;j++)bits[at++]=(v>>j)&1;
        };
        for(int i=0;i<K;i++) {
            append(read_word(f,4),16);
            auto row=read_word(f,8);
            for(int j=0;j<24;j++)read_word(f,1); // remaining wide-row pad bits unused
            append(row&((uint64_t(1)<<45)-1),45);append((row>>45)&65535,16);
            append(read_word(f,8),44);
            uint64_t pad=read_word(f,4);append(pad,16);pads.push_back(uint16_t(pad));
        }
        if(at!=n || f.get()!=EOF)throw std::runtime_error("bad file shape");
        NetIO io(role==ALICE?nullptr:"127.0.0.1",port,true);auto start=Clock::now();
        setup_semi_honest(&io,role);
        static_assert(sizeof(Bit)==sizeof(block128));std::vector<Bit> a(n),b(n);
        prot_exec->feed(reinterpret_cast<block128*>(a.data()),ALICE,bits.get(),n);
        prot_exec->feed(reinterpret_cast<block128*>(b.data()),BOB,bits.get(),n);
        io.flush();auto fed=Clock::now();
        auto input_sent=io.counter,input_received=io.received_counter;
        at=0;std::vector<Integer> outputs;
        for(int i=0;i<K;i++) {
            auto column=shared(a,b,at,16);
            auto threshold=shared(a,b,at,45);threshold.resize(46,false);
            auto alternate=shared(a,b,at,16);
            auto uniform=shared(a,b,at,44);uniform.resize(46,false);
            auto pad=shared(a,b,at,16);
            outputs.push_back(alternate.select(uniform<threshold,column)^pad);
        }
        io.flush();auto calculated=Clock::now();
        auto table_bytes=role==ALICE?io.counter-input_sent:io.received_counter-input_received;
        if(table_bytes%32)throw std::runtime_error("bad halfgate size");
        std::ofstream out(argv[4],std::ios::binary);
        for(int i=0;i<K;i++) {
            uint16_t v=uint16_t(outputs[i].reveal<uint32_t>(BOB));
            if(role==ALICE)v=pads[i];out.put(char(v&255));out.put(char(v>>8));
        }
        io.flush();auto done=Clock::now();out.close();
        std::ofstream m(argv[5]);m<<"{\"role\":"<<role<<",\"count\":201,\"and_gates\":"<<table_bytes/32
          <<",\"input_ms\":"<<ms(start,fed)<<",\"compute_ms\":"<<ms(fed,calculated)
          <<",\"release_ms\":"<<ms(calculated,done)<<",\"protocol_ms\":"<<ms(start,done)
          <<",\"sent\":"<<io.counter<<",\"received\":"<<io.received_counter<<"}\n";
        std::cout<<"ALIAS_BATCH_FINISH role="<<role<<std::endl;return 0;
    } catch(const std::exception& e) {std::cerr<<e.what()<<std::endl;return 1;}
}
