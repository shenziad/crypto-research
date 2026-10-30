#pragma once
// Standard arithmetic sharing, lexicographic comparison and inverse CDF.
// q masses sum to2^60; nonnegative exp31 values <=2^31. No new primitive.
#include "LinearOT/linear-ot.h"
#include "GC/emp-sh2pc.h"
#include <chrono>
#include <fstream>
#include <iostream>
#include <sys/random.h>

namespace private_cdf93 {
using U=unsigned __int128;
inline constexpr U mask93=(U(1)<<93)-1;
inline uint64_t mask(int n){return n==64?~uint64_t(0):(uint64_t(1)<<n)-1;}
inline U random93(){U v;if(getrandom(&v,sizeof(v),0)!=sizeof(v))throw std::runtime_error("random failure");return v&mask93;}
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
    if(n<1 || n>50304 || own_qmass.size()!=size_t(n))throw std::runtime_error("bad CDF shape");
    std::vector<uint64_t> low(n),high(n),plo(n),phi(n);U own_sum=0;
    for(int i=0;i<n;i++){
        if(own_qmass[i]>(uint64_t(1)<<60))throw std::runtime_error("bad q mass");
        if(role==ALICE && own_qmass[i])throw std::runtime_error("server must not have q");
        own_sum+=own_qmass[i];low[i]=own_qmass[i]&mask(30);high[i]=own_qmass[i]>>30;
    }
    if(role==BOB && own_sum!=(U(1)<<60))throw std::runtime_error("bad total q mass");
    // Fixedpublicchunks keep legacy transport calls bounded.
    for(int start=0;start<n;start+=4096){int count=std::min(4096,n-start);
        linear.hadamard_product(count,exp33.data()+start,low.data()+start,plo.data()+start,
            33,30,63,false,false,MultMode::Bob_has_B);
        linear.hadamard_product(count,exp33.data()+start,high.data()+start,phi.data()+start,
            33,31,64,false,false,MultMode::Bob_has_B);
    }
    words64(folder+"/weight_low_add63.bin",plo);words64(folder+"/weight_high_add64.bin",phi);
    std::vector<uint8_t> carry(n);std::vector<uint64_t> carry30(n);
    for(int start=0;start<n;start+=4096){int count=std::min(4096,n-start);
        linear.aux->wrap_computation(plo.data()+start,carry.data()+start,count,63);
        linear.aux->B2A(carry.data()+start,carry30.data()+start,count,30);
    }
    // Low63shares need a wrap correction to lift to93. High64shares shifted30
    // need none: their possible extra2^64 becomes2^94, zero modulo2^93.
    std::vector<U> prefix(n);U running=0;
    for(int i=0;i<n;i++){
        U w=(U(plo[i])-(U(carry30[i])<<63)+(U(phi[i])<<30))&mask93;
        running=(running+w)&mask93;prefix[i]=running;
    }
    words(folder+"/prefix_add93.bin",prefix);words64(folder+"/uniform_share.bin",{own_uniform});
    io.io->flush();io.io_rev->flush();io.io_GC->flush();
    if(!gc_ready)setup_semi_honest(io.io_GC,role);
    U pad=role==ALICE?random93():0;
    constexpr int input_size=93+64+93;bool own_bits[input_size];int at=0;
    for(int j=0;j<93;j++)own_bits[at++]=(running>>j)&1;
    for(int j=0;j<64;j++)own_bits[at++]=(own_uniform>>j)&1;
    for(int j=0;j<93;j++)own_bits[at++]=(pad>>j)&1;
    std::vector<Bit> a(input_size),b(input_size);
    static_assert(sizeof(Bit)==sizeof(block128));
    prot_exec->feed(reinterpret_cast<block128*>(a.data()),ALICE,own_bits,input_size);
    prot_exec->feed(reinterpret_cast<block128*>(b.data()),BOB,own_bits,input_size);
    Integer za(93,0,PUBLIC),zb(93,0,PUBLIC),u(157,0,PUBLIC),rho(93,0,PUBLIC);
    for(int j=0;j<93;j++){za[j]=a[j];zb[j]=b[j];rho[j]=a[157+j];}
    for(int j=0;j<64;j++)u[j]=a[93+j]^b[93+j];
    auto total=za+zb;total.resize(157,false);auto product=total*u;
    Integer r(93,0,PUBLIC);for(int j=0;j<93;j++)r[j]=product[64+j];
    auto masked=r-rho;bool revealed[93]={};
    prot_exec->reveal(revealed,BOB,reinterpret_cast<block128*>(masked.bits.data()),93);
    io.io_GC->flush();U own_r=pad;
    if(role==BOB){own_r=0;for(int j=0;j<93;j++)if(revealed[j])own_r|=U(1)<<j;}
    words(folder+"/r_add93.bin",{own_r});
    std::vector<uint8_t> gt_hi(n),eq_hi(n),gt_low(n),and_bits(n),negative(n);
    std::vector<uint64_t> hi(n),lo(n);std::vector<U> difference(n);
    for(int i=0;i<n;i++){
        difference[i]=(own_r-prefix[i])&mask93;
        U lower=difference[i]&((U(1)<<92)-1);
        if(role==BOB)lower^=(U(1)<<92)-1;
        hi[i]=uint64_t(lower>>64);lo[i]=uint64_t(lower);
    }
    for(int start=0;start<n;start+=4096){int count=std::min(4096,n-start);
        linear.aux->mill_and_eq->compare_with_eq(gt_hi.data()+start,eq_hi.data()+start,hi.data()+start,count,28,true);
        linear.aux->mill->compare(gt_low.data()+start,lo.data()+start,count,64,true);
        linear.aux->AND(eq_hi.data()+start,gt_low.data()+start,and_bits.data()+start,count);
    }
    uint16_t output_share=0;uint8_t previous=0;
    for(int i=0;i<n;i++){
        uint8_t bit=(uint8_t(difference[i]>>92)^gt_hi[i]^and_bits[i])&1;
        negative[i]=bit;uint8_t one_hot=bit^previous;previous=bit;
        output_share^=uint16_t(-int(one_hot))&uint16_t(i);
    }
    {std::ofstream f(folder+"/cdf_negative_xor.bin",std::ios::binary);f.write(reinterpret_cast<char*>(negative.data()),negative.size());}
    uint16_t token=output_share;
    if(role==ALICE){io.io_GC->send_data(&output_share,sizeof(output_share));io.io_GC->flush();}
    else{uint16_t other;io.io_GC->recv_data(&other,sizeof(other));token^=other;}
    std::ofstream out(folder+"/final_output.txt");if(role==BOB)out<<token<<"\n";
    return role==BOB?token:0;
}
} // namespace private_cdf93
