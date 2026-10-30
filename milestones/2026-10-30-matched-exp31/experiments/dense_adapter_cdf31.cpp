// Stronger dense-score component baseline, not a complete dense sampler.
// Reuse the frozen exact floor/PWL implementation without editing its source.
#define main frozen_bounded_adapter_reference_main
#include "bounded_adapter_gc.cpp"
#undef main
#include "LinearOT/linear-ot.h"
#include "utils/io_pack.h"
#include "OT/ot_pack.h"
#include "public_tanh_ot_chunked.h"
#include "private_weighted_cdf93.h"
#include "Math/math-functions.h"

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
        std::vector<uint64_t> qmass(rows);for(auto& v:qmass)v=read_word(input,8);
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
            auto scores=ot_pwl_chunked::evaluate(linear,raw_z);
            store(folder+"/delta_add44.bin",scores);
            MathFunctions math(role,&io,&ot);
            std::vector<uint64_t> neg(rows),exp33(rows);
            for(int i=0;i<rows;i++)neg[i]=(scores[i]-(role==ALICE?1<<20:0))&ring_mask(22);
            for(int start=0;start<rows;start+=512){int count=std::min(512,rows-start);
                std::cout<<"PUBLIC_EXP31_CHUNK begin="<<start<<" count="<<count<<std::endl;
                math.lookup_table_exp(count,neg.data()+start,exp33.data()+start,22,33,20,31);
            }
            store(folder+"/exp_add33.bin",exp33);
            uint64_t own_uniform;if(getrandom(&own_uniform,sizeof(own_uniform),0)!=sizeof(own_uniform))throw std::runtime_error("randomfailure");
            private_cdf93::sample(linear,io,exp33,qmass,own_uniform,folder,true);
            std::ofstream count(folder+"/nonlinear_gate_count.json");
            count<<"{\"and_gates\":null,\"backend\":\"standard_OT_PWL_exp31_CDF93\"}\n";
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
