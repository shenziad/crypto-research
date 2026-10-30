#pragma once
// Standard arithmetic sharing, lexicographic comparison and inverse CDF.
// q masses sum to2^62; E31<=2^31, trueZ<=2^93. No new primitive.
#include "LinearOT/linear-ot.h"
#include "GC/emp-sh2pc.h"
#include <chrono>
#include <fstream>
#include <iostream>
#include <sys/random.h>

namespace private_cdf95 {
using U=unsigned __int128;
inline constexpr U mask95=(U(1)<<95)-1;
inline uint64_t mask(int n){return n==64?~uint64_t(0):(uint64_t(1)<<n)-1;}
inline U random95(){U v;if(getrandom(&v,sizeof(v),0)!=sizeof(v))throw std::runtime_error("random failure");return v&mask95;}
inline void words(const std::string& path,const std::vector<U>& values){
    std::ofstream f(path,std::ios::binary);for(auto v:values)for(int j=0;j<16;j++)f.put(char(v>>(8*j)));f.close();
}
inline void words64(const std::string& path,const std::vector<uint64_t>& values){
    std::ofstream f(path,std::ios::binary);for(auto v:values)for(int j=0;j<8;j++)f.put(char(v>>(8*j)));f.close();
}

inline uint32_t sample(LinearOT& linear,sci::IOPack& io,std::vector<uint64_t>& exp33,
                      const std::vector<uint64_t>& own_qmass,uint64_t own_uniform,
                      const std::string& folder,bool gc_ready){
    using namespace sci;int role=linear.party,n=exp33.size();
    if(n<1 || n>151936 || own_qmass.size()!=size_t(n))throw std::runtime_error("bad CDF shape");
    std::vector<uint64_t> low(n),mid(n),high(n),plo(n),pmi(n),phi(n);U own_sum=0;
    for(int i=0;i<n;i++){
        if(own_qmass[i]>(uint64_t(1)<<62))throw std::runtime_error("bad q mass");
        if(role==ALICE && own_qmass[i])throw std::runtime_error("server must not have q");
        own_sum+=own_qmass[i];low[i]=own_qmass[i]&mask(30);
        mid[i]=(own_qmass[i]>>30)&mask(30);high[i]=own_qmass[i]>>60;
    }
    if(role==BOB && own_sum!=(U(1)<<62))throw std::runtime_error("bad total q mass");
    for(int start=0;start<n;start+=4096){int count=std::min(4096,n-start);
        linear.hadamard_product(count,exp33.data()+start,low.data()+start,plo.data()+start,
            33,30,63,false,false,MultMode::Bob_has_B);
        linear.hadamard_product(count,exp33.data()+start,mid.data()+start,pmi.data()+start,
            33,30,63,false,false,MultMode::Bob_has_B);
        linear.hadamard_product(count,exp33.data()+start,high.data()+start,phi.data()+start,
            33,3,36,false,false,MultMode::Bob_has_B);
    }
    words64(folder+"/weight_low_add63.bin",plo);words64(folder+"/weight_mid_add63.bin",pmi);
    words64(folder+"/weight_high_add36.bin",phi);
    std::vector<uint8_t> carry0(n),carry1(n);std::vector<uint64_t> carry32(n),carry2(n);
    for(int start=0;start<n;start+=4096){int count=std::min(4096,n-start);
        linear.aux->wrap_computation(plo.data()+start,carry0.data()+start,count,63);
        linear.aux->B2A(carry0.data()+start,carry32.data()+start,count,32);
        linear.aux->wrap_computation(pmi.data()+start,carry1.data()+start,count,63);
        linear.aux->B2A(carry1.data()+start,carry2.data()+start,count,2);
    }
    // Correct low63 lift into95 and middle63 shift30 into95. High36 shift60
    // needs no lift: extra2^36 becomes2^96, zero modulo2^95.
    std::vector<U> prefix(n);U running=0;
    for(int i=0;i<n;i++){
        U w=(U(plo[i])-(U(carry32[i])<<63)+(U(pmi[i])<<30)
             -(U(carry2[i])<<93)+(U(phi[i])<<60))&mask95;
        running=(running+w)&mask95;prefix[i]=running;
    }
    words(folder+"/prefix_add95.bin",prefix);words64(folder+"/uniform_share.bin",{own_uniform});
    io.io->flush();io.io_rev->flush();io.io_GC->flush();
    if(!gc_ready)setup_semi_honest(io.io_GC,role);
    U pad=role==ALICE?random95():0;
    constexpr int input_size=95+64+95;bool own_bits[input_size];int at=0;
    for(int j=0;j<95;j++)own_bits[at++]=(running>>j)&1;
    for(int j=0;j<64;j++)own_bits[at++]=(own_uniform>>j)&1;
    for(int j=0;j<95;j++)own_bits[at++]=(pad>>j)&1;
    std::vector<Bit> a(input_size),b(input_size);
    static_assert(sizeof(Bit)==sizeof(block128));
    prot_exec->feed(reinterpret_cast<block128*>(a.data()),ALICE,own_bits,input_size);
    prot_exec->feed(reinterpret_cast<block128*>(b.data()),BOB,own_bits,input_size);
    Integer za(95,0,PUBLIC),zb(95,0,PUBLIC),u(159,0,PUBLIC),rho(95,0,PUBLIC);
    for(int j=0;j<95;j++){za[j]=a[j];zb[j]=b[j];rho[j]=a[159+j];}
    for(int j=0;j<64;j++)u[j]=a[95+j]^b[95+j];
    auto total=za+zb;total.resize(159,false);auto product=total*u;
    Integer r(95,0,PUBLIC);for(int j=0;j<95;j++)r[j]=product[64+j];
    auto masked=r-rho;bool revealed[95]={};
    prot_exec->reveal(revealed,BOB,reinterpret_cast<block128*>(masked.bits.data()),95);
    io.io_GC->flush();U own_r=pad;
    if(role==BOB){own_r=0;for(int j=0;j<95;j++)if(revealed[j])own_r|=U(1)<<j;}
    words(folder+"/r_add95.bin",{own_r});
    std::vector<uint8_t> gt_hi(n),eq_hi(n),gt_low(n),and_bits(n),negative(n);
    std::vector<uint64_t> hi(n),lo(n);std::vector<U> difference(n);
    for(int i=0;i<n;i++){
        difference[i]=(own_r-prefix[i])&mask95;
        U lower=difference[i]&((U(1)<<94)-1);
        if(role==BOB)lower^=(U(1)<<94)-1;
        hi[i]=uint64_t(lower>>64);lo[i]=uint64_t(lower);
    }
    for(int start=0;start<n;start+=4096){int count=std::min(4096,n-start);
        linear.aux->mill_and_eq->compare_with_eq(gt_hi.data()+start,eq_hi.data()+start,hi.data()+start,count,30,true);
        linear.aux->mill->compare(gt_low.data()+start,lo.data()+start,count,64,true);
        linear.aux->AND(eq_hi.data()+start,gt_low.data()+start,and_bits.data()+start,count);
    }
    uint32_t output_share=0;uint8_t previous=0;
    for(int i=0;i<n;i++){
        uint8_t bit=(uint8_t(difference[i]>>94)^gt_hi[i]^and_bits[i])&1;
        negative[i]=bit;uint8_t one_hot=bit^previous;previous=bit;
        output_share^=uint32_t(-int(one_hot))&uint32_t(i);
    }
    {std::ofstream f(folder+"/cdf_negative_xor.bin",std::ios::binary);f.write(reinterpret_cast<char*>(negative.data()),negative.size());}
    uint32_t token=output_share;
    if(role==ALICE){io.io_GC->send_data(&output_share,sizeof(output_share));io.io_GC->flush();}
    else{uint32_t other;io.io_GC->recv_data(&other,sizeof(other));token^=other;}
    std::ofstream out(folder+"/final_output.txt");if(role==BOB)out<<token<<"\n";
    return role==BOB?token:0;
}
} // namespace private_cdf95
