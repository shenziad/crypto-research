// Stronger dense-score component baseline, not a complete dense sampler.
// Reuse the frozen exact floor/PWL implementation without editing its source.
#define main frozen_bounded_adapter_reference_main
#include "bounded_adapter_gc.cpp"
#undef main
#include "LinearOT/linear-ot.h"
#include "utils/io_pack.h"
#include "OT/ot_pack.h"
#include "public_tanh_compact.h"

static uint64_t ring_mask(int width) {return (uint64_t(1)<<width)-1;}
static uint64_t random44() {return ((uint64_t(random32())<<32)|random32())&ring_mask(44);}
static void store(const std::string& path,const std::vector<uint64_t>& values) {
    std::ofstream f(path,std::ios::binary);
    for(auto v:values)for(int j=0;j<8;j++)f.put(char((v>>(8*j))&255));
    f.close();if(!f)throw std::runtime_error("share write failure");
}
static std::array<uint64_t,6> counters(IOPack& io) {
    return {io.io->counter,io.io_rev->counter,io.io_GC->counter,
        io.io->received_counter,io.io_rev->received_counter,io.io_GC->received_counter};
}
static void flush(IOPack& io) {io.io->flush();io.io_rev->flush();io.io_GC->flush();}

int main(int argc,char** argv) {
    try {
        if(argc!=7)throw std::runtime_error("role port rows owninput ownfolder nonlinear");
        int role=std::stoi(argv[1]),port=std::stoi(argv[2]),rows=std::stoi(argv[3]);
        bool nonlinear=std::stoi(argv[6])!=0;
        if((role!=ALICE && role!=BOB) || rows<1 || rows>50304)throw std::runtime_error("bad shape");
        std::string folder=argv[5];std::ifstream input(argv[4],std::ios::binary);
        if(!input)throw std::runtime_error("no party input");
        std::vector<uint64_t> x(D),b(R*D),a(rows*R);
        for(auto& v:x) {v=read_word(input,8);if(v>>23)throw std::runtime_error("bad x23");}
        for(auto& v:b) {v=read_word(input,8);if(v>>18)throw std::runtime_error("bad b18");}
        for(auto& v:a) {v=read_word(input,8);if(v>>18)throw std::runtime_error("bad a18");}
        if(input.get()!=EOF)throw std::runtime_error("trailing input");
        auto start=Clock::now();
        IOPack io(role,port,"127.0.0.1");
        OTPack ot(&io,role,true);  // Fresh setup; never use OTPack::copy.
        LinearOT linear(role,&io,&ot);
        flush(io);auto setup=Clock::now();auto c_setup=counters(io);
        std::vector<uint64_t> b_products(R*D),raw_t(R,0);
        linear.matrix_multiplication(R,D,1,b.data(),x.data(),b_products.data(),18,23,41,
            true,true,false,MultMode::Alice_has_A);
        for(int j=0;j<R;j++)for(int k=0;k<D;k++)raw_t[j]=(raw_t[j]+b_products[j*D+k])&ring_mask(41);
        flush(io);auto bx_done=Clock::now();auto c_bx=counters(io);
        store(folder+"/raw_t_add41.bin",raw_t);
        setup_semi_honest(io.io_GC,role);
        std::vector<uint64_t> t(R);
        for(int j=0;j<R;j++) {
            Integer left(41,role==ALICE?raw_t[j]:0,ALICE),right(41,role==BOB?raw_t[j]:0,BOB);
            auto value=floor16(left+right);value.resize(26,true);
            uint64_t pad=role==ALICE?(random32()&ring_mask(26)):0;
            auto masked=value-Integer(26,pad,ALICE);
            auto revealed=masked.reveal<uint64_t>(BOB)&ring_mask(26);
            t[j]=role==ALICE?pad:revealed;
        }
        flush(io);auto bridge_done=Clock::now();auto c_bridge=counters(io);
        store(folder+"/t_add26.bin",t);
        std::vector<uint64_t> a_products(rows*R),raw_z(rows,0);
        linear.matrix_multiplication(rows,R,1,a.data(),t.data(),a_products.data(),18,26,44,
            true,true,false,MultMode::Alice_has_A);
        for(int i=0;i<rows;i++)for(int j=0;j<R;j++)raw_z[i]=(raw_z[i]+a_products[i*R+j])&ring_mask(44);
        flush(io);auto az_done=Clock::now();auto c_az=counters(io);
        store(folder+"/raw_z_add44.bin",raw_z);
        if(nonlinear) {
            // Batched additive input shares, plus server-only fresh44bit pads.
            // Output remains additive shares, never the dense scores themselves.
            size_t n=size_t(rows)*88,at=0;
            std::unique_ptr<bool[]> own(new bool[n]);std::vector<uint64_t> pads(rows,0);
            for(int i=0;i<rows;i++) {
                for(int j=0;j<44;j++)own[at++]=(raw_z[i]>>j)&1;
                if(role==ALICE)pads[i]=random44();
                for(int j=0;j<44;j++)own[at++]=(pads[i]>>j)&1;
            }
            std::vector<Bit> left(n),right(n);
            prot_exec->feed(reinterpret_cast<block128*>(left.data()),ALICE,own.get(),n);
            prot_exec->feed(reinterpret_cast<block128*>(right.data()),BOB,own.get(),n);
            flush(io);auto n_input=counters(io);at=0;
            std::vector<Integer> masked;
            masked.reserve(rows);
            for(int i=0;i<rows;i++) {
                Integer s(44,0,PUBLIC),c(44,0,PUBLIC),pad(44,0,PUBLIC);
                for(int j=0;j<44;j++,at++){s[j]=left[at];c[j]=right[at];}
                for(int j=0;j<44;j++,at++)pad[j]=left[at];
                auto z=floor16(s+c);auto delta=compact_pwl::tanh(z);delta.resize(44,true);
                masked.push_back(delta-pad);
            }
            flush(io);auto n_computed=counters(io);
            uint64_t table_bytes=role==ALICE?n_computed[2]-n_input[2]:n_computed[5]-n_input[5];
            if(table_bytes%32)throw std::runtime_error("bad halfgate size");
            std::vector<uint64_t> scores(rows);
            for(int i=0;i<rows;i++) {
                auto v=masked[i].reveal<uint64_t>(BOB)&ring_mask(44);
                scores[i]=role==ALICE?pads[i]:v;
            }
            store(folder+"/delta_add44.bin",scores);
            std::ofstream count(folder+"/nonlinear_gate_count.json");
            count<<"{\"and_gates\":"<<table_bytes/32<<",\"halfgate_table_bytes\":"<<table_bytes<<"}\n";
        }
        flush(io);auto done=Clock::now();auto c_end=counters(io);
        std::ofstream m(folder+"/metrics.json");
        auto array=[&](const std::array<uint64_t,6>& values) {
            m<<"[";for(int j=0;j<6;j++){if(j)m<<",";m<<values[j];}m<<"]";
        };
        m<<"{\"role\":"<<role<<",\"rows\":"<<rows<<",\"nonlinear\":"<<(nonlinear?"true":"false")
          <<",\"setup_ms\":"<<ms(start,setup)<<",\"bx_ms\":"<<ms(setup,bx_done)
          <<",\"bridge_ms\":"<<ms(bx_done,bridge_done)<<",\"az_ms\":"<<ms(bridge_done,az_done)
          <<",\"nonlinear_ms\":"<<ms(az_done,done)<<",\"protocol_ms\":"<<ms(start,done)
          <<",\"setup_counters\":";array(c_setup);m<<",\"bx_counters\":";array(c_bx);
        m<<",\"bridge_counters\":";array(c_bridge);m<<",\"az_counters\":";array(c_az);
        m<<",\"end_counters\":";array(c_end);m<<"}\n";
        std::cout<<"DENSE_SCORE_COMPONENT_FINISHED role="<<role<<std::endl;
        return 0;
    } catch(const std::exception& e) {std::cerr<<"dense adapter: "<<e.what()<<std::endl;return 1;}
}
