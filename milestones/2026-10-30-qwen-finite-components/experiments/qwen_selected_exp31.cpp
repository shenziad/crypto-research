// Matched E31 rejection head. Existing SCI operations, exact frozen PWL.
// A rows and indices arrive as XOR shares; no candidate is opened.
#define main frozen_bounded_adapter_reference_main
#include "bounded_adapter_gc.cpp"
#undef main
#include "public_tanh_ot_chunked.h"
#include "Math/math-functions.h"
#include "utils/io_pack.h"
#include "OT/ot_pack.h"
#include <filesystem>

static constexpr int QD=896;
static uint64_t mask_width(int n){return (uint64_t(1)<<n)-1;}
static void save64(const std::string& path,const std::vector<uint64_t>& values){
    std::ofstream f(path,std::ios::binary);for(auto v:values)for(int j=0;j<8;j++)f.put(char(v>>(8*j)));f.close();
}
int main(int argc,char** argv){
    try{
        if(argc!=8)throw std::runtime_error("role port count input output metrics token");
        int role=std::stoi(argv[1]),port=std::stoi(argv[2]),count=std::stoi(argv[3]);
        if((role!=1 && role!=2)||count<1||count>201||std::string(argv[7])!="token")throw std::runtime_error("badshape");
        std::string folder=std::filesystem::path(argv[5]).parent_path().string();
        std::ifstream f(argv[4],std::ios::binary);if(!f)throw std::runtime_error("noinput");
        std::vector<uint64_t> x(QD),b(R*QD),axor(count*R),indices(count),uniform(count);
        for(auto& v:x){v=read_word(f,4);if(v>>23)throw std::runtime_error("badx");
            int64_t signed_x=(v&(uint64_t(1)<<22))?int64_t(v)-(int64_t(1)<<23):int64_t(v);v=uint64_t(signed_x)&mask_width(23);}
        for(auto& v:b){v=read_word(f,4);if(v>>18)throw std::runtime_error("badb");}
        for(int i=0;i<count;i++){
            for(int j=0;j<R;j++){auto v=read_word(f,4);if(v>>18)throw std::runtime_error("badAshare");axor[i*R+j]=v;}
            indices[i]=read_word(f,4);if(indices[i]>>18)throw std::runtime_error("badindexshare");
            for(int j=0;j<38;j++){auto v=read_word(f,8);if(v>>44)throw std::runtime_error("badcoinshare");if(j==0)uniform[i]=v&mask_width(31);}
        }
        if(f.get()!=EOF)throw std::runtime_error("trailinginput");
        auto start=Clock::now();IOPack io(role,port,"127.0.0.1");OTPack ot(&io,role,true);LinearOT linear(role,&io,&ot);
        auto flush_all=[&](){io.io->flush();io.io_rev->flush();io.io_GC->flush();};
        std::vector<uint8_t> abit(size_t(count)*R*18);std::vector<uint64_t> ashare_bits(abit.size()),a(count*R,0);
        for(size_t i=0;i<axor.size();i++)for(int j=0;j<18;j++)abit[i*18+j]=(axor[i]>>j)&1;
        for(size_t p=0;p<abit.size();p+=4096){int n=std::min(size_t(4096),abit.size()-p);linear.aux->B2A(abit.data()+p,ashare_bits.data()+p,n,18);}
        for(size_t i=0;i<a.size();i++)for(int j=0;j<18;j++)a[i]=(a[i]+(ashare_bits[i*18+j]<<j))&mask_width(18);
        save64(folder+"/a_add18.bin",a);
        std::vector<uint64_t> products(R*QD),raw_t(R,0);
        linear.matrix_multiplication(R,QD,1,b.data(),x.data(),products.data(),18,23,41,true,true,false,MultMode::Alice_has_A);
        for(int i=0;i<R;i++)for(int j=0;j<QD;j++)raw_t[i]=(raw_t[i]+products[i*QD+j])&mask_width(41);
        save64(folder+"/raw_t_add41.bin",raw_t);flush_all();setup_semi_honest(io.io_GC,role);
        std::vector<uint64_t> t(R);
        for(int j=0;j<R;j++){
            Integer s(41,role==ALICE?raw_t[j]:0,ALICE),c(41,role==BOB?raw_t[j]:0,BOB);
            auto value=floor16(s+c);value.resize(26,true);
            auto pad=role==ALICE?random32()&mask_width(26):0;
            auto masked=value-Integer(26,pad,ALICE);auto output=masked.reveal<uint64_t>(BOB)&mask_width(26);
            t[j]=role==ALICE?pad:output;
        }
        save64(folder+"/t_add26.bin",t);flush_all();
        products.resize(count*R);std::vector<uint64_t> raw_z(count,0);
        linear.matrix_multiplication(count,R,1,a.data(),t.data(),products.data(),18,26,44,true,true,false,MultMode::None);
        for(int i=0;i<count;i++)for(int j=0;j<R;j++)raw_z[i]=(raw_z[i]+products[i*R+j])&mask_width(44);
        save64(folder+"/raw_z_add44.bin",raw_z);
        auto delta=ot_pwl_chunked::evaluate(linear,raw_z);save64(folder+"/delta_add44.bin",delta);
        MathFunctions math(role,&io,&ot);std::vector<uint64_t> neg(count),exp33(count);
        for(int i=0;i<count;i++)neg[i]=(delta[i]-(role==ALICE?1<<20:0))&mask_width(22);
        math.lookup_table_exp(count,neg.data(),exp33.data(),22,33,20,31);save64(folder+"/exp_add33.bin",exp33);flush_all();
        // Jointly uniform31bit acceptance, exact for the verified E31 law.
        size_t n=size_t(count)*(33+18+31),at=0;std::unique_ptr<bool[]> own(new bool[n]);
        for(int i=0;i<count;i++){
            for(int j=0;j<33;j++)own[at++]=(exp33[i]>>j)&1;
            for(int j=0;j<18;j++)own[at++]=(indices[i]>>j)&1;
            for(int j=0;j<31;j++)own[at++]=(uniform[i]>>j)&1;
        }
        std::vector<Bit> left(n),right(n);static_assert(sizeof(Bit)==sizeof(block128));
        prot_exec->feed(reinterpret_cast<block128*>(left.data()),ALICE,own.get(),n);
        prot_exec->feed(reinterpret_cast<block128*>(right.data()),BOB,own.get(),n);
        at=0;Integer result(32,262144,PUBLIC);Bit found(false,PUBLIC);
        for(int i=0;i<count;i++){
            Integer es(33,0,PUBLIC),ec(33,0,PUBLIC),index(32,0,PUBLIC),u(34,0,PUBLIC);
            for(int j=0;j<33;j++,at++){es[j]=left[at];ec[j]=right[at];}
            for(int j=0;j<18;j++,at++)index[j]=left[at]^right[at];
            for(int j=0;j<31;j++,at++)u[j]=left[at]^right[at];
            auto e=es+ec;e.resize(34,false);auto accepted=u<e;auto take=accepted&(!found);
            result=result.select(take,index);found=found|accepted;
        }
        auto token=result.reveal<uint32_t>(BOB);flush_all();auto done=Clock::now();
        std::ofstream out(argv[5]);if(role==BOB)out<<token<<"\n";
        uint64_t sent=io.io->counter+io.io_rev->counter+io.io_GC->counter;
        uint64_t received=io.io->received_counter+io.io_rev->received_counter+io.io_GC->received_counter;
        std::ofstream metrics(argv[6]);metrics<<"{\"role\":"<<role<<",\"count\":"<<count
         <<",\"protocol_ms\":"<<ms(start,done)<<",\"sent\":"<<sent<<",\"received\":"<<received
         <<",\"profile\":\"Qwen896_index18_E31_private_rejection\"}\n";
        std::cout<<"QWEN_SELECTED_E31_HEAD_FINISHED role="<<role<<std::endl;return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}
}
