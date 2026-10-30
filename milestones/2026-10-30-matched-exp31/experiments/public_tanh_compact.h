#pragma once
// Standard public one-hot coefficient selection and carry-save multiplication.
// Exact same finite PWL function, no new primitive or precision change.
#include "GC/emp-sh2pc.h"
#include <stdexcept>
#include <vector>

namespace compact_pwl {
inline constexpr int nodes[17]={0,256816,484565,666002,798589,889490,949117,987104,
    1010856,1025535,1034540,1040040,1043391,1045428,1046665,1047417,1047873};

template<class B> std::vector<B> one_hot4(const std::vector<B>& bits,B zero) {
    std::vector<B> values{!bits[0],bits[0]};
    for(int k=1;k<4;k++) {
        std::vector<B> next(2*values.size(),zero);
        for(size_t j=0;j<values.size();j++) {
            B high=values[j]&bits[k];
            next[j]=values[j]^high;next[j+values.size()]=high;
        }
        values=std::move(next);
    }
    return values;
}

template<class B> std::vector<B> coefficient(const std::vector<B>& selector,
                                           int width,bool slope,B zero) {
    std::vector<B> out(width,zero);
    for(int j=0;j<16;j++) {
        unsigned value=slope?nodes[j+1]-nodes[j]:nodes[j];
        for(int k=0;k<width;k++)if((value>>k)&1)out[k]=out[k]^selector[j];
    }
    return out;
}

template<class B> std::vector<B> multiply14x18(const std::vector<B>& a,
                                            const std::vector<B>& b,B zero) {
    if(a.size()!=14 || b.size()!=18)throw std::runtime_error("bad public multiplier shape");
    std::vector<std::vector<B>> columns(33);
    for(int i=0;i<14;i++)for(int j=0;j<18;j++)columns[i+j].push_back(a[i]&b[j]);
    for(int k=0;k<32;k++) {
        while(columns[k].size()>=3) {
            B x=columns[k].back();columns[k].pop_back();
            B y=columns[k].back();columns[k].pop_back();
            B z=columns[k].back();columns[k].pop_back();
            B sum=x^y^z;
            B carry=x^((x^y)&(x^z));
            columns[k].push_back(sum);columns[k+1].push_back(carry);
        }
    }
    std::vector<B> out(32,zero);B carry=zero;
    for(int k=0;k<32;k++) {
        B x=columns[k].empty()?zero:B(columns[k][0]);
        B y=columns[k].size()<2?zero:B(columns[k][1]);
        out[k]=x^y^carry;
        if(k<31)carry=x^((x^y)&(x^carry));
    }
    return out;
}

inline sci::Integer tanh(const sci::Integer& z) {
    using namespace sci;
    auto magnitude=z.abs();Bit zero(false,PUBLIC);
    std::vector<Bit> interval(4),remainder(14);
    for(int k=0;k<4;k++)interval[k]=magnitude[14+k];
    for(int k=0;k<14;k++)remainder[k]=magnitude[k];
    auto hot=one_hot4(interval,zero);
    auto node=coefficient(hot,21,false,zero);
    auto slope=coefficient(hot,18,true,zero);
    auto product=multiply14x18(remainder,slope,zero);
    Integer value(21,0,PUBLIC),increment(21,0,PUBLIC);
    for(int k=0;k<21;k++)value[k]=node[k];
    for(int k=0;k<18;k++)increment[k]=product[14+k];
    auto positive=value+increment;positive.resize(22,false);
    auto saturated=magnitude>=Integer(z.size(),4*65536,PUBLIC);
    positive=positive.select(saturated,Integer(22,1<<20,PUBLIC));
    return positive.select(z[z.size()-1],-positive);
}
} // namespace compact_pwl
