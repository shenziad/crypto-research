// Standard two-coordinate DCF, with arbitrary public linear query weights.
// Includes unchanged reference primitive and its upstream MIT notice.
#include "dcf_sparse_native.cpp"
struct LinearQuery { uint64_t x,w[2]; };

static void aggregate_tree(const uint8_t *key,int bits,int party,int level,const Seed &seed,
    uint8_t control,const LinearQuery *queries,int lo,int hi,uint64_t &out) {
    constexpr int step=33;__m128i keys[11];schedule(seed,keys);
    if(level==bits) {
        uint64_t weights[2]={0,0};for(int j=lo;j<hi;j++){weights[0]+=queries[j].w[0];weights[1]+=queries[j].w[1];}
        if(!(weights[0]|weights[1]))return;
        uint64_t value[2],final[2];_mm_storeu_si128((__m128i*)value,enc_dcf(keys,5));std::memcpy(final,key+16+step*bits,16);
        for(int c=0;c<2;c++) {
            uint64_t v=value[c]+(control?final[c]:0);out+=weights[c]*(party?uint64_t(0-v):v);
        }
        return;
    }
    Seed children[2],cw;uint8_t t[16];
    _mm_storeu_si128((__m128i*)children[0].v,enc_dcf(keys,0));_mm_storeu_si128((__m128i*)children[1].v,enc_dcf(keys,1));
    _mm_storeu_si128((__m128i*)t,enc_dcf(keys,2));std::memcpy(cw.v,key+16+step*level,16);uint8_t tcw=key[16+step*level+16];
    int mid=lo;while(mid<hi && !((queries[mid].x>>(bits-1-level))&1))mid++;
    for(int branch=0;branch<2;branch++) {
        int begin=branch?mid:lo,end=branch?hi:mid;if(begin==end)continue;
        uint64_t weights[2]={0,0};for(int j=begin;j<end;j++){weights[0]+=queries[j].w[0];weights[1]+=queries[j].w[1];}
        if(weights[0]|weights[1]) {
            uint64_t value[2],correction[2];_mm_storeu_si128((__m128i*)value,enc_dcf(keys,3+branch));
            std::memcpy(correction,key+16+step*level+17,16);
            for(int c=0;c<2;c++){uint64_t v=value[c]+(control?correction[c]:0);out+=weights[c]*(party?uint64_t(0-v):v);}
        }
        Seed next=children[branch];uint8_t nt=(t[0]>>branch)&1;if(control){next=sxor(next,cw);nt^=(tcw>>branch)&1;}
        aggregate_tree(key,bits,party,level+1,next,nt,queries,begin,end,out);
    }
}

static void linear_tree(const uint8_t *key,int bits,int party,int level,const Seed &seed,
    uint8_t control,const uint64_t *sum,const LinearQuery *queries,int lo,int hi,uint64_t &out) {
    constexpr int step=33;
    if(level==bits) {
        uint64_t final[2],v[2];dcf_final(seed,v);std::memcpy(final,key+16+step*bits,16);
        for(int c=0;c<2;c++) {
            uint64_t term=v[c]+(control?final[c]:0);v[c]=sum[c]+(party?uint64_t(0-term):term);
        }
        for(int j=lo;j<hi;j++)out+=queries[j].w[0]*v[0]+queries[j].w[1]*v[1];
        return;
    }
    Seed child[2],cw;uint8_t t[2];uint64_t values[2][2],vcw[2];dcf_expand(seed,child,t,values);
    std::memcpy(cw.v,key+16+step*level,16);uint8_t tcw=key[16+step*level+16];
    std::memcpy(vcw,key+16+step*level+17,16);
    int mid=lo;while(mid<hi && !((queries[mid].x>>(bits-1-level))&1))mid++;
    for(int b=0;b<2;b++) {
        int begin=b?mid:lo,end=b?hi:mid;if(begin==end)continue;
        uint64_t next_sum[2];
        for(int c=0;c<2;c++) {
            uint64_t v=values[b][c]+(control?vcw[c]:0);next_sum[c]=sum[c]+(party?uint64_t(0-v):v);
        }
        Seed next=child[b];uint8_t nt=t[b];if(control){next=sxor(next,cw);nt^=(tcw>>b)&1;}
        linear_tree(key,bits,party,level+1,next,nt,next_sum,queries,begin,end,out);
    }
}

extern "C" int dcf_linear_queries(int bits,int n,int party,int count,int cached,
    const uint8_t *keys,const uint64_t *queries,const uint64_t *weights,uint64_t *out) {
    if(bits<1 || bits>63 || n<1 || party<0 || party>1 || count<1)return -1;
    int size=16+bits*33+16;std::vector<LinearQuery> ordered(count);
    for(int i=0;i<n;i++) {
        const uint8_t *key=keys+i*size;out[i]=0;
        if(!cached) {
            for(int j=0;j<count;j++) {
                uint64_t x=queries[i*count+j],v[2];dcf_eval_impl<uint64_t>(bits,2,1,party,key,&x,v);
                out[i]+=weights[(i*count+j)*2]*v[0]+weights[(i*count+j)*2+1]*v[1];
            }
        } else {
            for(int j=0;j<count;j++)ordered[j]={queries[i*count+j],{weights[(i*count+j)*2],weights[(i*count+j)*2+1]}};
            std::sort(ordered.begin(),ordered.end(),[](const LinearQuery &a,const LinearQuery &b){return a.x<b.x;});
            Seed seed;std::memcpy(seed.v,key,16);uint64_t sum[2]={0,0};
            if(cached==2)aggregate_tree(key,bits,party,0,seed,party,ordered.data(),0,count,out[i]);
            else linear_tree(key,bits,party,0,seed,party,sum,ordered.data(),0,count,out[i]);
        }
    }
    return 0;
}
