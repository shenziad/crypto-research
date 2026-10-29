// Same matched endpoint tree; separate public classifier/body coordinates.
#include "dcf_selector_native.cpp"
extern "C" int dcf_dither_spline(int bits,int m,int n,int party,int mode,const uint8_t *keys,
    const uint64_t *body,const uint64_t *classifier,const uint64_t *starts,const uint64_t *slopes,const uint64_t *intercepts,uint64_t *out) {
    if(bits<1||bits>63||m<1||m>64||n<1||party<0||party>1||mode<0||mode>3)return -1;
    for(int j=1;j<m;j++)if(starts[j]<=starts[j-1])return -1;
    int dim=mode==0?1:mode==1?2:mode==2?4:2*m,size=16+bits*(17+8*dim)+8*dim;
    uint64_t mask=(uint64_t(1)<<bits)-1;std::vector<SplineQuery> queries(m);
    for(int row=0;row<n;row++){
        uint64_t h=body[row],hc=classifier[row]&mask,cutoff=(hc-1)&mask;
        int last=int(std::upper_bound(starts,starts+m,cutoff)-starts)-1;if(last<0)return -1;
        for(int j=0;j<m;j++){int i=(last-j+m)%m;queries[j]={(hc-starts[i]-1)&mask,i};}
        const uint8_t *key=keys+size_t(row)*size;Seed seed;std::memcpy(seed.v,key,16);uint64_t *result=out+2*row;result[0]=result[1]=0;
        if(mode==0)coefficient_tree<false>(key,bits,m,party,h,0,seed,party,queries.data(),0,m,slopes,intercepts,result);
        else if(mode==1)common_endpoint_tree<true>(key,bits,m,party,h,0,seed,party,queries.data(),0,m,slopes,intercepts,result[0]);
        else if(mode==2)coefficient_tree<true>(key,bits,m,party,h,0,seed,party,queries.data(),0,m,slopes,intercepts,result);
        else common_endpoint_tree<false>(key,bits,m,party,h,0,seed,party,queries.data(),0,m,slopes,intercepts,result[0]);
    }
    return 0;
}
