// Variable public payload dimension, standard DCF correction algebra.
// Counter layout extends the original dim1/2 backend exactly: child0/1,
// controls2, alternating left/right value blocks3+, followed by final blocks.
#include "dpf_aes_native.cpp"
#include <algorithm>

static inline uint64_t vector_word(const __m128i *schedule,int counter,int parity) {
    uint64_t block[2];_mm_storeu_si128((__m128i*)block,enc_dcf(schedule,counter));return block[parity];
}
static inline void expand_wide(const Seed &seed,int dim,Seed child[2],uint8_t t[2],uint64_t *left,uint64_t *right) {
    __m128i keys[11];schedule(seed,keys);uint8_t control[16];
    _mm_storeu_si128((__m128i*)child[0].v,enc_dcf(keys,0));_mm_storeu_si128((__m128i*)child[1].v,enc_dcf(keys,1));
    _mm_storeu_si128((__m128i*)control,enc_dcf(keys,2));t[0]=control[0]&1;t[1]=(control[0]>>1)&1;
    for(int j=0;j<(dim+1)/2;j++) {
        uint64_t a[2],b[2];_mm_storeu_si128((__m128i*)a,enc_dcf(keys,3+2*j));_mm_storeu_si128((__m128i*)b,enc_dcf(keys,4+2*j));
        for(int k=0;k<2 && 2*j+k<dim;k++){left[2*j+k]=a[k];right[2*j+k]=b[k];}
    }
}
static inline void final_wide(const Seed &seed,int dim,uint64_t *out) {
    __m128i keys[11];schedule(seed,keys);int blocks=(dim+1)/2;
    for(int j=0;j<blocks;j++) {
        uint64_t v[2];_mm_storeu_si128((__m128i*)v,enc_dcf(keys,3+2*blocks+j));
        for(int k=0;k<2 && 2*j+k<dim;k++)out[2*j+k]=v[k];
    }
}

extern "C" int dcf_wide_generate(int bits,int dim,int n,const uint64_t *alpha,const uint64_t *beta,
    const uint8_t *roots,uint8_t *out0,uint8_t *out1) {
    if(bits<1 || bits>63 || dim<1 || dim>128 || n<1)return -1;
    int step=17+8*dim,size=16+step*bits+8*dim;
    std::vector<uint64_t> along(dim),value(4*dim),vcw(dim),f0(dim),f1(dim),final(dim);
    for(int row=0;row<n;row++) {
        Seed seeds[2];uint8_t control[2]={0,1};uint8_t *out[2]={out0+row*size,out1+row*size};std::fill(along.begin(),along.end(),0);
        for(int p=0;p<2;p++){std::memcpy(seeds[p].v,roots+row*32+p*16,16);std::memcpy(out[p],seeds[p].v,16);}
        for(int level=0;level<bits;level++) {
            int keep=(alpha[row]>>(bits-1-level))&1,lose=keep^1;Seed children[2][2];uint8_t t[2][2];
            for(int p=0;p<2;p++)expand_wide(seeds[p],dim,children[p],t[p],value.data()+(2*p)*dim,value.data()+(2*p+1)*dim);
            for(int c=0;c<dim;c++) {
                uint64_t base=0-along[c]-value[lose*dim+c]+value[(2+lose)*dim+c]+(keep?beta[row*dim+c]:0);
                vcw[c]=control[1]?uint64_t(0-base):base;
                along[c]+=uint64_t(0-value[(2+keep)*dim+c])+value[keep*dim+c]+(control[1]?uint64_t(0-vcw[c]):vcw[c]);
            }
            Seed cw=sxor(children[0][lose],children[1][lose]);uint8_t tl=t[0][0]^t[1][0]^keep^1,tr=t[0][1]^t[1][1]^keep;
            for(int p=0;p<2;p++) {
                std::memcpy(out[p]+16+step*level,cw.v,16);out[p][16+step*level+16]=tl|(tr<<1);
                std::memcpy(out[p]+16+step*level+17,vcw.data(),8*dim);
                seeds[p]=children[p][keep];uint8_t nt=t[p][keep];if(control[p]){seeds[p]=sxor(seeds[p],cw);nt^=keep?tr:tl;}control[p]=nt;
            }
        }
        final_wide(seeds[0],dim,f0.data());final_wide(seeds[1],dim,f1.data());
        for(int c=0;c<dim;c++){final[c]=f1[c]-f0[c]-along[c];if(control[1])final[c]=0-final[c];}
        for(int p=0;p<2;p++)std::memcpy(out[p]+16+step*bits,final.data(),8*dim);
    }
    return 0;
}

static void select_one(const uint8_t *key,int bits,int dim,int party,uint64_t x,const int *coords,int count,uint64_t *out) {
    int step=17+8*dim,blocks=(dim+1)/2;Seed seed;std::memcpy(seed.v,key,16);uint8_t control=party;
    std::fill(out,out+count,0);
    for(int level=0;level<bits;level++) {
        __m128i keys[11];schedule(seed,keys);Seed children[2],cw;uint8_t t[16];
        _mm_storeu_si128((__m128i*)children[0].v,enc_dcf(keys,0));_mm_storeu_si128((__m128i*)children[1].v,enc_dcf(keys,1));
        _mm_storeu_si128((__m128i*)t,enc_dcf(keys,2));int b=(x>>(bits-1-level))&1;
        uint64_t cached_values[64][2];bool valid[64]={false};
        for(int j=0;j<count;j++) {
            int c=coords[j],block=c/2;if(!valid[block]){_mm_storeu_si128((__m128i*)cached_values[block],enc_dcf(keys,3+2*block+b));valid[block]=true;}
            uint64_t correction;std::memcpy(&correction,key+16+step*level+17+8*c,8);
            uint64_t v=cached_values[block][c%2]+(control?correction:0);out[j]+=party?uint64_t(0-v):v;
        }
        std::memcpy(cw.v,key+16+step*level,16);uint8_t tcw=key[16+step*level+16];seed=children[b];uint8_t nt=(t[0]>>b)&1;
        if(control){seed=sxor(seed,cw);nt^=(tcw>>b)&1;}control=nt;
    }
    __m128i keys[11];schedule(seed,keys);uint64_t cached_values[64][2];bool valid[64]={false};
    for(int j=0;j<count;j++) {
        int c=coords[j],block=c/2;if(!valid[block]){_mm_storeu_si128((__m128i*)cached_values[block],enc_dcf(keys,3+2*blocks+block));valid[block]=true;}
        uint64_t correction;std::memcpy(&correction,key+16+step*bits+8*c,8);
        uint64_t v=cached_values[block][c%2]+(control?correction:0);out[j]+=party?uint64_t(0-v):v;
    }
}
extern "C" int dcf_wide_select(int bits,int dim,int n,int party,int count,const uint8_t *keys,
    const uint64_t *x,const int *coords,uint64_t *out) {
    if(bits<1 || bits>63 || dim<1 || dim>128 || count<1 || count>128)return -1;
    int size=16+bits*(17+8*dim)+8*dim;
    for(int j=0;j<count;j++)if(coords[j]<0 || coords[j]>=dim)return -1;
    for(int i=0;i<n;i++)select_one(keys+i*size,bits,dim,party,x[i],coords,count,out+i*count);
    return 0;
}

struct SplineQuery {uint64_t x;int interval;};
static inline uint64_t endpoint_value(const uint8_t *key,const __m128i *keys,int bits,int dim,int level,
    int branch,bool final,int positive,int negative,uint64_t h,uint8_t control,int party) {
    if(positive==negative)return 0;
    int step=17+8*dim,blocks=(dim+1)/2;uint64_t total=0;
    for(int term=0;term<2;term++) {
        int poly=term?negative:positive;uint64_t value[2],cw[2];
        int counter=final?3+2*blocks+poly:3+2*poly+branch;
        _mm_storeu_si128((__m128i*)value,enc_dcf(keys,counter));
        std::memcpy(cw,key+16+step*(final?bits:level)+(final?0:17)+16*poly,16);
        uint64_t v=h*(value[0]+(control?cw[0]:0))+value[1]+(control?cw[1]:0);
        if(term)total-=v;else total+=v;
    }
    return party?uint64_t(0-total):total;
}
static void spline_tree(const uint8_t *key,int bits,int dim,int party,uint64_t h,int level,
    const Seed &seed,uint8_t control,const SplineQuery *queries,int lo,int hi,uint64_t &out,bool endpoints) {
    int step=17+8*dim,blocks=(dim+1)/2;
    __m128i keys[11];schedule(seed,keys);
    if(level==bits) {
        if(endpoints) {
            int positive=(queries[hi-1].interval+dim/2-1)%(dim/2),negative=queries[lo].interval;
            out+=endpoint_value(key,keys,bits,dim,level,0,true,positive,negative,h,control,party);return;
        }
        uint64_t weights[128]={0};
        for(int j=lo;j<hi;j++) {
            int i=queries[j].interval,prev=(i+dim/2-1)%(dim/2);weights[2*prev]+=h;weights[2*prev+1]++;
            weights[2*i]-=h;weights[2*i+1]--;
        }
        for(int b=0;b<blocks;b++)if(weights[2*b] || (2*b+1<dim && weights[2*b+1])) {
            uint64_t value[2];_mm_storeu_si128((__m128i*)value,enc_dcf(keys,3+2*blocks+b));
            for(int j=0;j<2 && 2*b+j<dim;j++)if(weights[2*b+j]) {
                uint64_t cw;std::memcpy(&cw,key+16+step*bits+8*(2*b+j),8);
                uint64_t v=value[j]+(control?cw:0);out+=weights[2*b+j]*(party?uint64_t(0-v):v);
            }
        }
        return;
    }
    Seed children[2],cw;uint8_t tc[16];
    _mm_storeu_si128((__m128i*)children[0].v,enc_dcf(keys,0));_mm_storeu_si128((__m128i*)children[1].v,enc_dcf(keys,1));
    _mm_storeu_si128((__m128i*)tc,enc_dcf(keys,2));std::memcpy(cw.v,key+16+step*level,16);uint8_t tcw=key[16+step*level+16];
    int mid=lo;while(mid<hi && !((queries[mid].x>>(bits-1-level))&1))mid++;
    for(int branch=0;branch<2;branch++) {
        int begin=branch?mid:lo,end=branch?hi:mid;if(begin==end)continue;
        if(endpoints) {
            int positive=(queries[end-1].interval+dim/2-1)%(dim/2),negative=queries[begin].interval;
            out+=endpoint_value(key,keys,bits,dim,level,branch,false,positive,negative,h,control,party);
            Seed next=children[branch];uint8_t nt=(tc[0]>>branch)&1;if(control){next=sxor(next,cw);nt^=(tcw>>branch)&1;}
            spline_tree(key,bits,dim,party,h,level+1,next,nt,queries,begin,end,out,endpoints);continue;
        }
        uint64_t weights[128]={0};
        for(int j=begin;j<end;j++) {
            int i=queries[j].interval,prev=(i+dim/2-1)%(dim/2);weights[2*prev]+=h;weights[2*prev+1]++;
            weights[2*i]-=h;weights[2*i+1]--;
        }
        for(int b=0;b<blocks;b++)if(weights[2*b] || (2*b+1<dim && weights[2*b+1])) {
            uint64_t value[2];_mm_storeu_si128((__m128i*)value,enc_dcf(keys,3+2*b+branch));
            for(int j=0;j<2 && 2*b+j<dim;j++)if(weights[2*b+j]) {
                uint64_t correction;std::memcpy(&correction,key+16+step*level+17+8*(2*b+j),8);
                uint64_t v=value[j]+(control?correction:0);out+=weights[2*b+j]*(party?uint64_t(0-v):v);
            }
        }
        Seed next=children[branch];uint8_t nt=(tc[0]>>branch)&1;if(control){next=sxor(next,cw);nt^=(tcw>>branch)&1;}
        spline_tree(key,bits,dim,party,h,level+1,next,nt,queries,begin,end,out,endpoints);
    }
}

extern "C" int dcf_wide_spline(int bits,int m,int n,int party,int cached,const uint8_t *keys,
    const uint64_t *opened,const uint64_t *starts,uint64_t *out) {
    if(bits<1 || bits>63 || m<1 || 2*m>128 || n<1 || party<0 || party>1)return -1;
    for(int i=1;i<m;i++)if(starts[i]<=starts[i-1])return -1;
    int dim=2*m,size=16+bits*(17+8*dim)+8*dim;uint64_t mask=(uint64_t(1)<<bits)-1;
    std::vector<SplineQuery> queries(m);
    for(int row=0;row<n;row++) {
        const uint8_t *key=keys+row*size;uint64_t h=opened[row];out[row]=0;
        if(!cached) {
            for(int i=0;i<m;i++) {
                uint64_t x=(h-starts[i]-1)&mask,v[4];int prev=(i+m-1)%m,coords[4]={2*prev,2*prev+1,2*i,2*i+1};
                select_one(key,bits,dim,party,x,coords,4,v);out[row]+=h*(v[0]-v[2])+v[1]-v[3];
            }
        } else {
            for(int i=0;i<m;i++)queries[i]={(h-starts[i]-1)&mask,i};
            std::sort(queries.begin(),queries.end(),[](const SplineQuery &a,const SplineQuery &b){return a.x<b.x;});
            Seed seed;std::memcpy(seed.v,key,16);
            spline_tree(key,bits,dim,party,h,0,seed,party,queries.data(),0,m,out[row],cached==2);
        }
    }
    return 0;
}

// Both payload layouts use this same minimal public-prefix traversal. Sorted
// queries occur in reverse cyclic interval order, so query weights telescope
// to two endpoint polynomials. No scan of the full payload vector is needed.
template<bool Basis> static void common_endpoint_tree(const uint8_t *key,int bits,int m,int party,uint64_t h,int level,
    const Seed &seed,uint8_t control,const SplineQuery *queries,int lo,int hi,
    const uint64_t *slopes,const uint64_t *intercepts,uint64_t &out) {
    int dim=Basis?2:2*m,step=17+8*dim;__m128i keys[11];schedule(seed,keys);
    auto add=[&](int begin,int end,int branch,bool final) {
        int positive=(queries[end-1].interval+m-1)%m,negative=queries[begin].interval;
        if constexpr(Basis) {
            uint64_t dm=slopes[positive]-slopes[negative],dc=intercepts[positive]-intercepts[negative];
            uint64_t w0=dm*h+dc,w1=0-dm;if(!(w0|w1))return;
            uint64_t value[2],cw[2];_mm_storeu_si128((__m128i*)value,enc_dcf(keys,final?5:3+branch));
            std::memcpy(cw,key+16+step*(final?bits:level)+(final?0:17),16);
            uint64_t v=w0*(value[0]+(control?cw[0]:0))+w1*(value[1]+(control?cw[1]:0));
            out+=party?uint64_t(0-v):v;
        } else out+=endpoint_value(key,keys,bits,dim,level,branch,final,positive,negative,h,control,party);
    };
    if(level==bits){add(lo,hi,0,true);return;}
    Seed child[2],cw;uint8_t t[16];
    _mm_storeu_si128((__m128i*)child[0].v,enc_dcf(keys,0));_mm_storeu_si128((__m128i*)child[1].v,enc_dcf(keys,1));
    _mm_storeu_si128((__m128i*)t,enc_dcf(keys,2));std::memcpy(cw.v,key+16+step*level,16);uint8_t tcw=key[16+step*level+16];
    int mid=lo;while(mid<hi && !((queries[mid].x>>(bits-1-level))&1))mid++;
    for(int b=0;b<2;b++) {
        int begin=b?mid:lo,end=b?hi:mid;if(begin==end)continue;add(begin,end,b,false);
        Seed next=child[b];uint8_t nt=(t[0]>>b)&1;if(control){next=sxor(next,cw);nt^=(tcw>>b)&1;}
        common_endpoint_tree<Basis>(key,bits,m,party,h,level+1,next,nt,queries,begin,end,slopes,intercepts,out);
    }
}

extern "C" int dcf_spline_endpoint(int bits,int m,int n,int party,int basis,const uint8_t *keys,
    const uint64_t *opened,const uint64_t *starts,const uint64_t *slopes,const uint64_t *intercepts,uint64_t *out) {
    if(bits<1 || bits>63 || m<1 || 2*m>128 || n<1 || party<0 || party>1)return -1;
    for(int i=1;i<m;i++)if(starts[i]<=starts[i-1])return -1;
    int dim=basis?2:2*m,size=16+bits*(17+8*dim)+8*dim;uint64_t mask=(uint64_t(1)<<bits)-1;
    std::vector<SplineQuery> queries(m);
    for(int row=0;row<n;row++) {
        uint64_t h=opened[row],cutoff=(h-1)&mask;
        int last=int(std::upper_bound(starts,starts+m,cutoff)-starts)-1;
        if(last<0)return -1;
        for(int j=0;j<m;j++){int i=(last-j+m)%m;queries[j]={(h-starts[i]-1)&mask,i};}
        const uint8_t *key=keys+row*size;Seed seed;std::memcpy(seed.v,key,16);out[row]=0;
        if(basis)common_endpoint_tree<true>(key,bits,m,party,h,0,seed,party,queries.data(),0,m,slopes,intercepts,out[row]);
        else common_endpoint_tree<false>(key,bits,m,party,h,0,seed,party,queries.data(),0,m,slopes,intercepts,out[row]);
    }
    return 0;
}
