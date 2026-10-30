// Conditional-domain finite adapter arithmetic and optional acceptance/select.
// Existing SCI semi-honest GC. Both inputs are party-local XOR shares; the
// public backbone input is client-owned and B is server-owned (zero other share).
#include "GC/emp-sh2pc.h"
#include <array>
#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <sys/random.h>
#include <vector>

using namespace sci;
using Clock=std::chrono::steady_clock;
constexpr int D=128,R=8,P=44,F=20,L=19;
constexpr int NODES[17]={0,256816,484565,666002,798589,889490,949117,987104,
    1010856,1025535,1034540,1040040,1043391,1045428,1046665,1047417,1047873};

static uint64_t read_word(std::ifstream& f,int bytes) {
    uint64_t out=0;
    for(int j=0;j<bytes;j++) {
        int c=f.get();if(c==EOF)throw std::runtime_error("short input");
        out|=uint64_t(uint8_t(c))<<(8*j);
    }
    return out;
}

static Integer shared(const std::vector<Bit>& a,const std::vector<Bit>& b,size_t& at,int width) {
    Integer out(width,0,PUBLIC);
    for(int j=0;j<width;j++,at++)out[j]=a[at]^b[at];
    return out;
}

static Integer floor16(const Integer& input) {
    // Explicit signed-floor slice. SCI's operator>> fills high bits with zero;
    // retaining its old width would therefore be wrong for negative inputs.
    Integer out(input.size()-16,0,PUBLIC);
    for(int j=0;j<out.size();j++)out[j]=input[j+16];
    return out;
}

static Integer pwl_tanh(const Integer& z) {
    auto magnitude=z.abs();
    std::vector<Integer> values,slopes;
    for(int j=0;j<16;j++) {
        values.emplace_back(21,NODES[j],PUBLIC);
        slopes.emplace_back(19,NODES[j+1]-NODES[j],PUBLIC);
    }
    for(int bit=0;bit<4;bit++) {
        std::vector<Integer> next_v,next_s;
        for(size_t j=0;j<values.size();j+=2) {
            next_v.push_back(values[j].select(magnitude[14+bit],values[j+1]));
            next_s.push_back(slopes[j].select(magnitude[14+bit],slopes[j+1]));
        }
        values=std::move(next_v);slopes=std::move(next_s);
    }
    Integer remainder(34,0,PUBLIC);
    for(int j=0;j<14;j++)remainder[j]=magnitude[j];
    slopes[0].resize(34,false);
    auto product=remainder*slopes[0];
    Integer increment(21,0,PUBLIC);
    for(int j=0;j<20;j++)increment[j]=product[j+14];
    auto positive=values[0]+increment;
    positive.resize(22,false);
    auto saturated=magnitude>=Integer(z.size(),4*65536,PUBLIC);
    positive=positive.select(saturated,Integer(22,1<<F,PUBLIC));
    return positive.select(z[z.size()-1],-positive);
}

static Bit accept(const Integer& delta,const std::vector<Bit>& a,
                  const std::vector<Bit>& b,size_t& at) {
    auto extended=delta;extended.resize(P+2,true);
    auto threshold=(Integer(P+2,1<<F,PUBLIC)-extended)<<(P-F-1);
    Bit accepted(true,PUBLIC);
    for(int chain=0;chain<2;chain++) {
        auto previous=threshold;
        Bit active(true,PUBLIC),parity(false,PUBLIC);
        for(int j=0;j<L;j++) {
            auto uniform=shared(a,b,at,P);uniform.resize(P+2,false);
            active=active & (uniform<previous);
            parity=parity ^ active;
            previous=uniform;
        }
        accepted=accepted & ((!active)&(!parity));
    }
    return accepted;
}

static uint32_t random32() {
    uint32_t out;
    if(getrandom(&out,sizeof(out),0)!=sizeof(out))throw std::runtime_error("random failure");
    return out;
}

static double ms(Clock::time_point a,Clock::time_point b) {
    return std::chrono::duration<double,std::milli>(b-a).count();
}

int main(int argc,char** argv) {
    try {
        if(argc!=8)throw std::runtime_error("role port count partyinput output metrics mode");
        int role=std::stoi(argv[1]),port=std::stoi(argv[2]),count=std::stoi(argv[3]);
        std::string mode=argv[7];bool conformance=mode=="shares";
        if((mode!="shares" && mode!="token") || (role!=ALICE && role!=BOB) || count<1 || count>201)
            throw std::runtime_error("bad public shape");
        // The same input format is used in both public output modes.
        size_t n=D*22+R*D*18+size_t(count)*(R*18+16+2*L*P),at=0;
        std::unique_ptr<bool[]> local(new bool[n]);
        std::ifstream input(argv[4],std::ios::binary);
        if(!input)throw std::runtime_error("no party input");
        auto append=[&](uint64_t v,int width) {
            if(v>>width)throw std::runtime_error("bad word width");
            for(int j=0;j<width;j++)local[at++]=(v>>j)&1;
        };
        for(int j=0;j<D;j++)append(read_word(input,4),22);
        for(int j=0;j<R*D;j++)append(read_word(input,4),18);
        for(int i=0;i<count;i++) {
            for(int j=0;j<R;j++)append(read_word(input,4),18);
            append(read_word(input,4),16);
            for(int j=0;j<2*L;j++)append(read_word(input,8),P);
        }
        if(at!=n || input.get()!=EOF)throw std::runtime_error("bad input size");
        NetIO io(role==ALICE?nullptr:"127.0.0.1",port,true);
        auto start=Clock::now();setup_semi_honest(&io,role);auto setup=Clock::now();
        static_assert(sizeof(Bit)==sizeof(block128));
        std::vector<Bit> left(n),right(n);
        prot_exec->feed(reinterpret_cast<block128*>(left.data()),ALICE,local.get(),n);
        prot_exec->feed(reinterpret_cast<block128*>(right.data()),BOB,local.get(),n);
        io.flush();auto fed=Clock::now();
        auto input_sent=io.counter,input_received=io.received_counter;
        auto before=circ_exec->num_and();at=0;
        std::vector<Integer> x,t;
        for(int j=0;j<D;j++)x.push_back(shared(left,right,at,22));
        for(int j=0;j<R;j++) {
            Integer total(41,0,PUBLIC);
            for(int k=0;k<D;k++) {
                auto coefficient=shared(left,right,at,18);coefficient.resize(41,true);
                auto coordinate=x[k];coordinate.resize(41,true);
                total=total+coordinate*coefficient;
            }
            t.push_back(floor16(total));
        }
        std::vector<Integer> zs,deltas;
        Integer result(32,65536,PUBLIC);Bit found(false,PUBLIC);
        for(int i=0;i<count;i++) {
            Integer total(44,0,PUBLIC);
            for(int j=0;j<R;j++) {
                auto coefficient=shared(left,right,at,18);coefficient.resize(44,true);
                auto coordinate=t[j];coordinate.resize(44,true);
                total=total+coordinate*coefficient;
            }
            auto z=floor16(total);auto delta=pwl_tanh(z);
            zs.push_back(z);deltas.push_back(delta);
            auto index=shared(left,right,at,16);index.resize(32,false);
            // Even in conformance mode, evaluate the exact acceptance circuit.
            auto accepted=accept(delta,left,right,at);
            auto take=accepted & (!found);result=result.select(take,index);
            found=found | accepted;
        }
        if(at!=n)throw std::runtime_error("wire layout mismatch");
        io.flush();auto computed=Clock::now();
        auto hash_counter=circ_exec->num_and()-before;
        auto table_bytes=role==ALICE?io.counter-input_sent:io.received_counter-input_received;
        if(table_bytes%32)throw std::runtime_error("nonintegral halfgate table size");
        auto arithmetic_and_count=table_bytes/32;
        std::ofstream output(argv[5],std::ios::binary);
        if(conformance) {
            std::vector<Integer> released=t;
            released.insert(released.end(),zs.begin(),zs.end());
            released.insert(released.end(),deltas.begin(),deltas.end());
            for(auto value:released) {
                uint32_t mask=role==ALICE?random32():0;
                value.resize(32,true);
                auto masked=value^Integer(32,mask,ALICE);
                auto share=masked.reveal<uint32_t>(BOB);
                uint32_t own=role==ALICE?mask:share;
                for(int j=0;j<4;j++)output.put(char((own>>(8*j))&255));
            }
        } else {
            uint32_t token=result.reveal<uint32_t>(BOB);
            if(role==BOB)output<<token<<"\n";
        }
        io.flush();auto done=Clock::now();output.close();
        std::ofstream metrics(argv[6]);
        metrics<<"{\"role\":"<<role<<",\"count\":"<<count<<",\"mode\":\""<<mode
          <<"\",\"and_gates_before_release\":"<<arithmetic_and_count
          <<",\"upstream_hash_key_counter_delta\":"<<hash_counter
          <<",\"halfgate_table_bytes_before_release\":"<<table_bytes
          <<",\"setup_ms\":"<<ms(start,setup)<<",\"input_ms\":"<<ms(setup,fed)
          <<",\"compute_ms\":"<<ms(fed,computed)<<",\"release_ms\":"<<ms(computed,done)
          <<",\"protocol_ms\":"<<ms(start,done)<<",\"sent\":"<<io.counter
          <<",\"received\":"<<io.received_counter<<",\"io_direction_changes\":"<<io.num_rounds<<"}\n";
        std::cout<<"BOUNDED_ADAPTER_FINISHED role="<<role<<std::endl;
        return 0;
    } catch(const std::exception& e) {
        std::cerr<<"adapter error: "<<e.what()<<std::endl;return 1;
    }
}
