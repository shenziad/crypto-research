// Exhaust all remainders for every actual public PWL slope, plus decoder.
#include "public_tanh_compact.h"
#include <cstdint>
#include <iostream>

static std::vector<bool> bits(uint64_t v,int width) {
    std::vector<bool> out(width);for(int j=0;j<width;j++)out[j]=(v>>j)&1;return out;
}
static uint64_t value(const std::vector<bool>& b) {
    uint64_t v=0;for(size_t j=0;j<b.size();j++)if(b[j])v|=uint64_t(1)<<j;return v;
}
int main() {
    uint64_t cases=0;
    for(int j=0;j<16;j++) {
        auto hot=compact_pwl::one_hot4(bits(j,4),false);
        if(value(hot)!=(uint64_t(1)<<j))throw std::runtime_error("decoder mismatch");
        auto node=compact_pwl::coefficient(hot,21,false,false);
        auto slope=compact_pwl::coefficient(hot,18,true,false);
        if(value(node)!=unsigned(compact_pwl::nodes[j]) ||
           value(slope)!=unsigned(compact_pwl::nodes[j+1]-compact_pwl::nodes[j]))
            throw std::runtime_error("coefficient mismatch");
        for(unsigned remainder=0;remainder<(1u<<14);remainder++) {
            auto product=compact_pwl::multiply14x18(bits(remainder,14),slope,false);
            if(value(product)!=uint64_t(remainder)*value(slope))throw std::runtime_error("product mismatch");
            cases++;
        }
    }
    for(auto a:{0u,1u,(1u<<14)-1})for(auto b:{0u,1u,(1u<<18)-1}) {
        if(value(compact_pwl::multiply14x18(bits(a,14),bits(b,18),false))!=uint64_t(a)*b)
            throw std::runtime_error("boundary product mismatch");
        cases++;
    }
    std::cout<<"{\"all_passed\":true,\"actual_slope_remainder_cases\":"<<cases
             <<",\"decoder_cases\":16}"<<std::endl;
    return 0;
}
