#pragma once
// Exact frozen PWL via existing SCI truncation/MSB/mux/public LUT/multiply.
// Standard arithmetic implementation, not a new cryptographic construction.
#include "LinearOT/linear-ot.h"
#include "public_tanh_compact.h"
#include <sys/random.h>

namespace ot_pwl {
inline uint64_t mask(int width) {return (uint64_t(1)<<width)-1;}
inline uint64_t random_word(int width) {
    uint64_t value;
    if(getrandom(&value,sizeof(value),0)!=sizeof(value))throw std::runtime_error("random failure");
    return value&mask(width);
}

inline std::vector<uint64_t> lookup(LinearOT& linear,std::vector<uint64_t>& index,bool slope) {
    int n=index.size(),width=slope?18:21;
    std::vector<uint64_t> output(n);
    if(linear.party==sci::ALICE) {
        std::vector<uint64_t> table(size_t(n)*16);
        std::vector<uint64_t*> pointers(n);
        for(int i=0;i<n;i++) {
            output[i]=random_word(width);pointers[i]=table.data()+size_t(i)*16;
            for(int j=0;j<16;j++) {
                int k=(index[i]+j)&15;
                uint64_t v=slope?compact_pwl::nodes[k+1]-compact_pwl::nodes[k]:compact_pwl::nodes[k];
                pointers[i][j]=(v-output[i])&mask(width);
            }
        }
        linear.aux->lookup_table<uint64_t>(pointers.data(),nullptr,nullptr,n,4,width);
    } else {
        linear.aux->lookup_table<uint64_t>(nullptr,index.data(),output.data(),n,4,width);
        for(auto& value:output)value&=mask(width);
    }
    return output;
}

inline std::vector<uint64_t> evaluate(LinearOT& linear,std::vector<uint64_t>& raw_z) {
    int n=raw_z.size();
    std::vector<uint64_t> z(n),negative_part(n),magnitude(n),bin(n),remainder(n),difference(n);
    std::vector<uint8_t> sign(n),below_saturation(n),saturated(n);
    linear.trunc->truncate_and_reduce(n,raw_z.data(),z.data(),16,44);
    linear.aux->MSB(z.data(),sign.data(),n,28);
    linear.aux->multiplexer(sign.data(),z.data(),negative_part.data(),n,28,28);
    for(int i=0;i<n;i++) {
        magnitude[i]=(z[i]-2*negative_part[i])&mask(28);
        remainder[i]=magnitude[i]&mask(14);
        difference[i]=(magnitude[i]-(linear.party==sci::ALICE?4*65536:0))&mask(28);
    }
    linear.aux->MSB(difference.data(),below_saturation.data(),n,28);
    for(int i=0;i<n;i++)saturated[i]=below_saturation[i]^(linear.party==sci::ALICE?1:0);
    linear.trunc->truncate_and_reduce(n,magnitude.data(),bin.data(),14,28);
    for(auto& value:bin)value&=15;
    auto node=lookup(linear,bin,false);auto slope=lookup(linear,bin,true);
    std::vector<uint64_t> product(n),increment(n),node22(n),increment22(n),positive(n),clamp_product(n);
    linear.hadamard_product(n,remainder.data(),slope.data(),product.data(),14,18,32,false,false);
    linear.trunc->truncate_and_reduce(n,product.data(),increment.data(),14,32);
    linear.xt->z_extend(n,node.data(),node22.data(),21,22);
    linear.xt->z_extend(n,increment.data(),increment22.data(),18,22);
    for(int i=0;i<n;i++) {
        positive[i]=(node22[i]+increment22[i])&mask(22);
        difference[i]=((linear.party==sci::ALICE?1<<20:0)-positive[i])&mask(22);
    }
    linear.aux->multiplexer(saturated.data(),difference.data(),clamp_product.data(),n,22,22);
    for(int i=0;i<n;i++)positive[i]=(positive[i]+clamp_product[i])&mask(22);
    linear.aux->multiplexer(sign.data(),positive.data(),negative_part.data(),n,22,22);
    std::vector<uint64_t> delta(n),extended(n);
    for(int i=0;i<n;i++)delta[i]=(positive[i]-2*negative_part[i])&mask(22);
    linear.xt->s_extend(n,delta.data(),extended.data(),22,44);
    return extended;
}
} // namespace ot_pwl
