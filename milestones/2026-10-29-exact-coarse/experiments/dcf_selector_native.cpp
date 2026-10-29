// Compact public coefficient selection and weighted public-basis affine gates.
// Both reuse unchanged variable-dimension DCF generation and its PRG layout.
#include "dcf_wide_native.cpp"

template<bool Weighted> static void coefficient_tree(const uint8_t *key,int bits,int m,int party,uint64_t h,int level,
    const Seed &seed,uint8_t control,const SplineQuery *queries,int lo,int hi,
    const uint64_t *slopes,const uint64_t *intercepts,uint64_t *out) {
    constexpr int dim=Weighted?4:1,step=17+8*dim,blocks=(dim+1)/2;
    __m128i keys[11];schedule(seed,keys);
    auto add=[&](int begin,int end,int branch,bool final) {
        int positive=(queries[end-1].interval+m-1)%m,negative=queries[begin].interval;
        uint64_t dm=slopes[positive]-slopes[negative],dc=intercepts[positive]-intercepts[negative];
        if(!(dm|dc))return;
        if constexpr(Weighted) {
            uint64_t w0=dm*h+dc,w1=0-dm;
            for(int pair=0;pair<2;pair++) {
                uint64_t v[2],cw[2];int counter=final?3+2*blocks+pair:3+2*pair+branch;
                _mm_storeu_si128((__m128i*)v,enc_dcf(keys,counter));
                std::memcpy(cw,key+16+step*(final?bits:level)+(final?0:17)+16*pair,16);
                uint64_t value=w0*(v[0]+(control?cw[0]:0))+w1*(v[1]+(control?cw[1]:0));
                out[pair]+=party?uint64_t(0-value):value;
            }
        } else {
            uint64_t v[2],cw;_mm_storeu_si128((__m128i*)v,enc_dcf(keys,final?5:3+branch));
            std::memcpy(&cw,key+16+step*(final?bits:level)+(final?0:17),8);
            uint64_t value=v[0]+(control?cw:0);if(party)value=0-value;
            out[0]+=dm*value;out[1]+=dc*value;
        }
    };
    if(level==bits){add(lo,hi,0,true);return;}
    Seed children[2],cw;uint8_t t[16];
    _mm_storeu_si128((__m128i*)children[0].v,enc_dcf(keys,0));_mm_storeu_si128((__m128i*)children[1].v,enc_dcf(keys,1));
    _mm_storeu_si128((__m128i*)t,enc_dcf(keys,2));std::memcpy(cw.v,key+16+step*level,16);uint8_t tcw=key[16+step*level+16];
    int mid=lo;while(mid<hi && !((queries[mid].x>>(bits-1-level))&1))mid++;
    for(int b=0;b<2;b++) {
        int begin=b?mid:lo,end=b?hi:mid;if(begin==end)continue;add(begin,end,b,false);
        Seed next=children[b];uint8_t nt=(t[0]>>b)&1;if(control){next=sxor(next,cw);nt^=(tcw>>b)&1;}
        coefficient_tree<Weighted>(key,bits,m,party,h,level+1,next,nt,queries,begin,end,slopes,intercepts,out);
    }
}

extern "C" int dcf_public_coefficient(int bits,int m,int n,int party,int weighted,const uint8_t *keys,
    const uint64_t *opened,const uint64_t *starts,const uint64_t *slopes,const uint64_t *intercepts,uint64_t *out) {
    if(bits<1 || bits>63 || m<1 || m>64 || n<1 || party<0 || party>1)return -1;
    for(int i=1;i<m;i++)if(starts[i]<=starts[i-1])return -1;
    int dim=weighted?4:1,size=16+bits*(17+8*dim)+8*dim;uint64_t mask=(uint64_t(1)<<bits)-1;
    std::vector<SplineQuery> queries(m);
    for(int row=0;row<n;row++) {
        uint64_t h=opened[row],cutoff=(h-1)&mask;int last=int(std::upper_bound(starts,starts+m,cutoff)-starts)-1;
        if(last<0)return -1;
        for(int j=0;j<m;j++){int i=(last-j+m)%m;queries[j]={(h-starts[i]-1)&mask,i};}
        const uint8_t *key=keys+row*size;Seed seed;std::memcpy(seed.v,key,16);out[2*row]=out[2*row+1]=0;
        if(weighted)coefficient_tree<true>(key,bits,m,party,h,0,seed,party,queries.data(),0,m,slopes,intercepts,out+2*row);
        else coefficient_tree<false>(key,bits,m,party,h,0,seed,party,queries.data(),0,m,slopes,intercepts,out+2*row);
    }
    return 0;
}
