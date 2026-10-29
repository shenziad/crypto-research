// Public shifted-query weighted DCF evaluation. Standard DCF arithmetic is
// included unchanged; this file only shares public query-prefix computation.
#include "dpf_aes_native.cpp"
#include <algorithm>
struct Query { uint64_t x,w; };

static void weighted_tree(const uint8_t *key,int bits,int dim,int party,int level,
    const Seed &seed,uint8_t control,const uint64_t *sum,const Query *queries,
    int lo,int hi,uint64_t *out) {
    const int step=17+8*dim;
    if(level==bits) {
        uint64_t final[2]={0,0},value[2];dcf_final(seed,value);
        std::memcpy(final,key+16+step*bits,8*dim);
        for(int c=0;c<dim;c++) {
            uint64_t v=value[c]+(control?final[c]:0);
            uint64_t result=sum[c]+(party?uint64_t(0-v):v);
            for(int j=lo;j<hi;j++)out[c]+=queries[j].w*result;
        }
        return;
    }
    Seed child[2],cw;uint8_t t[2];uint64_t values[2][2],vcw[2]={0,0};
    dcf_expand(seed,child,t,values);
    std::memcpy(cw.v,key+16+step*level,16);
    uint8_t tcw=key[16+step*level+16];std::memcpy(vcw,key+16+step*level+17,8*dim);
    int mid=lo;while(mid<hi && !((queries[mid].x>>(bits-1-level))&1))mid++;
    for(int branch=0;branch<2;branch++) {
        int begin=branch?mid:lo,end=branch?hi:mid;if(begin==end)continue;
        uint64_t next_sum[2];
        for(int c=0;c<dim;c++) {
            uint64_t v=values[branch][c]+(control?vcw[c]:0);
            next_sum[c]=sum[c]+(party?uint64_t(0-v):v);
        }
        Seed next=child[branch];uint8_t next_control=t[branch];
        if(control){next=sxor(next,cw);next_control^=(tcw>>branch)&1;}
        weighted_tree(key,bits,dim,party,level+1,next,next_control,next_sum,queries,begin,end,out);
    }
}

extern "C" int dcf_sparse_weighted(int bits,int dim,int n,int party,int count,int cached,
    const uint8_t *keys,const uint64_t *opened,const uint64_t *thresholds,const uint64_t *weights,uint64_t *out) {
    if(bits<1 || bits>63 || dim<1 || dim>2 || n<1 || count<1 || party<0 || party>1)return -1;
    const int size=16+bits*(17+8*dim)+8*dim;const uint64_t mask=(uint64_t(1)<<bits)-1;
    std::vector<Query> queries(count);
    for(int i=0;i<n;i++) {
        const uint8_t *key=keys+i*size;uint64_t *row=out+i*dim;
        for(int c=0;c<dim;c++)row[c]=0;
        if(!cached) {
            for(int j=0;j<count;j++) {
                uint64_t x=(opened[i]-thresholds[j])&mask,value[2];
                dcf_eval_impl<uint64_t>(bits,dim,1,party,key,&x,value);
                for(int c=0;c<dim;c++)row[c]+=weights[j]*value[c];
            }
        } else {
            for(int j=0;j<count;j++)queries[j]={(opened[i]-thresholds[j])&mask,weights[j]};
            std::sort(queries.begin(),queries.end(),[](const Query &a,const Query &b){return a.x<b.x;});
            Seed seed;std::memcpy(seed.v,key,16);uint64_t sum[2]={0,0};
            weighted_tree(key,bits,dim,party,0,seed,party,sum,queries.data(),0,count,row);
        }
    }
    return 0;
}
