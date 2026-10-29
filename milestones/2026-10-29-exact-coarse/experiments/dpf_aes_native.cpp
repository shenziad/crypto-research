// Research BGI-style DPF backend. AES-128 with seed as key is the PRG:
// counter blocks 0/1 give children, block 2 gives two control bits, block 3
// converts a leaf to one/two uint64 group elements. Not an audited library.
#include <immintrin.h>
#include <cstdint>
#include <cstring>
#include <vector>

struct Seed { uint8_t v[16]; };
static inline Seed sxor(const Seed &a,const Seed &b) {
    Seed c; _mm_storeu_si128((__m128i*)c.v,_mm_xor_si128(_mm_loadu_si128((const __m128i*)a.v),_mm_loadu_si128((const __m128i*)b.v))); return c;
}
static inline void schedule(const Seed &s,__m128i *keys) {
    __m128i a=_mm_loadu_si128((const __m128i*)s.v);keys[0]=a;
#define STEP(I,R) { __m128i t=_mm_aeskeygenassist_si128(a,R);t=_mm_shuffle_epi32(t,0xff);__m128i z=_mm_slli_si128(a,4);a=_mm_xor_si128(a,z);z=_mm_slli_si128(z,4);a=_mm_xor_si128(a,z);z=_mm_slli_si128(z,4);a=_mm_xor_si128(a,z);a=_mm_xor_si128(a,t);keys[I]=a; }
    STEP(1,0x01);STEP(2,0x02);STEP(3,0x04);STEP(4,0x08);STEP(5,0x10);
    STEP(6,0x20);STEP(7,0x40);STEP(8,0x80);STEP(9,0x1b);STEP(10,0x36);
#undef STEP
}
static inline __m128i enc(const __m128i *keys,uint64_t counter) {
    __m128i state=_mm_set_epi64x(0x4c4c4d4450467631LL,(long long)counter);
    state=_mm_xor_si128(state,keys[0]);
    for(int j=1;j<10;j++)state=_mm_aesenc_si128(state,keys[j]);
    return _mm_aesenclast_si128(state,keys[10]);
}
static inline void expand(const Seed &s,Seed &left,Seed &right,uint8_t &tl,uint8_t &tr) {
    __m128i keys[11];schedule(s,keys);
    _mm_storeu_si128((__m128i*)left.v,enc(keys,0));
    _mm_storeu_si128((__m128i*)right.v,enc(keys,1));
    uint8_t t[16];_mm_storeu_si128((__m128i*)t,enc(keys,2));tl=t[0]&1;tr=(t[0]>>1)&1;
}
template<class Word> static inline void convert(const Seed &s,Word out[2]) {
    __m128i keys[11];schedule(s,keys);uint8_t block[16];
    _mm_storeu_si128((__m128i*)block,enc(keys,3));std::memcpy(out,block,2*sizeof(Word));
}

template<class Word> int generate_impl(int bits,int dim,int n,const uint8_t *alpha,const uint64_t *beta,
    const uint8_t *roots,uint8_t *out0,uint8_t *out1) {
    if(bits!=8 || dim<1 || dim>2 || n<1)return -1;
    const int keysize=16+17*bits+sizeof(Word)*dim;
    for(int i=0;i<n;i++) {
        uint8_t *out[2]={out0+i*keysize,out1+i*keysize};Seed seeds[2];uint8_t controls[2]={0,1};
        for(int b=0;b<2;b++){std::memcpy(seeds[b].v,roots+i*32+b*16,16);std::memcpy(out[b],seeds[b].v,16);}
        for(int level=0;level<bits;level++) {
            int keep=(alpha[i]>>(bits-1-level))&1,lose=keep^1;
            Seed child[2][2];uint8_t t[2][2];
            for(int b=0;b<2;b++)expand(seeds[b],child[b][0],child[b][1],t[b][0],t[b][1]);
            Seed cw=sxor(child[0][lose],child[1][lose]);
            uint8_t tl=t[0][0]^t[1][0]^keep^1,tr=t[0][1]^t[1][1]^keep;
            for(int b=0;b<2;b++) {
                std::memcpy(out[b]+16+17*level,cw.v,16);out[b][16+17*level+16]=tl|(tr<<1);
                seeds[b]=child[b][keep];uint8_t nt=t[b][keep];
                if(controls[b]){seeds[b]=sxor(seeds[b],cw);nt^=keep?tr:tl;} controls[b]=nt;
            }
        }
        Word v0[2],v1[2];convert(seeds[0],v0);convert(seeds[1],v1);
        for(int j=0;j<dim;j++) {
            Word cw=Word(beta[i*dim+j]-v0[j]+v1[j]);if(controls[1])cw=Word(0-cw);
            for(int b=0;b<2;b++)std::memcpy(out[b]+16+17*bits+sizeof(Word)*j,&cw,sizeof(Word));
        }
    }
    return 0;
}

template<class Word> int lookup_impl(int bits,int dim,int n,int party,const uint8_t *keys,
    const uint8_t *ahat,const uint64_t *table,uint64_t *outputs,int table_stride) {
    if(bits!=8 || dim<1 || dim>2 || n<1 || party<0 || party>1)return -1;
    if(table_stride!=0 && table_stride!=256)return -1;
    const int keysize=16+17*bits+sizeof(Word)*dim;
    Seed a[256],b[256];uint8_t ta[256],tb[256];
    for(int item=0;item<n;item++) {
        const uint8_t *key=keys+item*keysize;std::memcpy(a[0].v,key,16);ta[0]=party;
        Seed *in=a,*out=b;uint8_t *tin=ta,*tout=tb;
        int count=1;
        for(int level=0;level<bits;level++) {
            Seed cw;std::memcpy(cw.v,key+16+17*level,16);uint8_t tcw=key[16+17*level+16];
            for(int j=0;j<count;j++) {
                expand(in[j],out[2*j],out[2*j+1],tout[2*j],tout[2*j+1]);
                if(tin[j])for(int branch=0;branch<2;branch++) {
                    out[2*j+branch]=sxor(out[2*j+branch],cw);tout[2*j+branch]^=(tcw>>branch)&1;
                }
            }
            Seed *tmp=in;in=out;out=tmp;uint8_t *tt=tin;tin=tout;tout=tt;count*=2;
        }
        Word final[2]={0,0};std::memcpy(final,key+16+17*bits,dim*sizeof(Word));
        Word sum[2]={0,0};
        for(int j=0;j<256;j++) {
            Word value[2];convert(in[j],value);Word coefficient=Word(table[item*table_stride+((int(ahat[item])-j)&255)]);
            for(int c=0;c<dim;c++) {
                if(tin[j])value[c]+=final[c];if(party)value[c]=0-value[c];
                sum[c]=Word(uint64_t(sum[c])+uint64_t(coefficient)*uint64_t(value[c]));
            }
        }
        for(int c=0;c<dim;c++)outputs[item*dim+c]=sum[c];
    }
    return 0;
}

extern "C" int dpf_generate_batch(int bits,int dim,int n,const uint8_t *alpha,const uint64_t *beta,
    const uint8_t *roots,uint8_t *out0,uint8_t *out1,int group_bits) {
    if(group_bits==32)return generate_impl<uint32_t>(bits,dim,n,alpha,beta,roots,out0,out1);
    if(group_bits==16)return generate_impl<uint16_t>(bits,dim,n,alpha,beta,roots,out0,out1);
    if(group_bits==64)return generate_impl<uint64_t>(bits,dim,n,alpha,beta,roots,out0,out1);
    return -1;
}

extern "C" int dpf_lookup_batch(int bits,int dim,int n,int party,const uint8_t *keys,
    const uint8_t *ahat,const uint64_t *table,uint64_t *outputs,int table_stride,int group_bits) {
    if(group_bits==32)return lookup_impl<uint32_t>(bits,dim,n,party,keys,ahat,table,outputs,table_stride);
    if(group_bits==16)return lookup_impl<uint16_t>(bits,dim,n,party,keys,ahat,table,outputs,table_stride);
    if(group_bits==64)return lookup_impl<uint64_t>(bits,dim,n,party,keys,ahat,table,outputs,table_stride);
    return -1;
}

/* DCF correction-value algorithm adapted from EzPC/FSS/src/dcf.cpp.
Authors: Deepak Kumaraswamy, Kanav Gupta
Copyright (c) 2022 Microsoft Research
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
Local adaptation: explicit control bits and seed-key AES PRG, packed byte keys,
OS roots supplied by Python, independent of upstream runtime/random sampler.
*/
static inline __m128i enc_dcf(const __m128i *keys,uint64_t counter) {
    __m128i state=_mm_set_epi64x(0x4c4c4d4443467631LL,(long long)counter);
    state=_mm_xor_si128(state,keys[0]);
    for(int j=1;j<10;j++)state=_mm_aesenc_si128(state,keys[j]);
    return _mm_aesenclast_si128(state,keys[10]);
}
template<class Word> static inline void dcf_expand(const Seed &s,Seed child[2],uint8_t t[2],Word v[2][2]) {
    __m128i keys[11];schedule(s,keys);uint8_t block[16];
    _mm_storeu_si128((__m128i*)child[0].v,enc_dcf(keys,0));
    _mm_storeu_si128((__m128i*)child[1].v,enc_dcf(keys,1));
    _mm_storeu_si128((__m128i*)block,enc_dcf(keys,2));t[0]=block[0]&1;t[1]=(block[0]>>1)&1;
    _mm_storeu_si128((__m128i*)block,enc_dcf(keys,3));std::memcpy(v[0],block,2*sizeof(Word));
    _mm_storeu_si128((__m128i*)block,enc_dcf(keys,4));std::memcpy(v[1],block,2*sizeof(Word));
}
template<class Word> static inline void dcf_final(const Seed &s,Word out[2]) {
    __m128i keys[11];schedule(s,keys);uint8_t block[16];
    _mm_storeu_si128((__m128i*)block,enc_dcf(keys,5));std::memcpy(out,block,2*sizeof(Word));
}

template<class Word> int dcf_generate_impl(int bits,int dim,int n,const uint64_t *alpha,const uint64_t *beta,
    const uint8_t *roots,uint8_t *out0,uint8_t *out1) {
    if(bits<1 || bits>63 || dim<1 || dim>2 || n<1)return -1;
    const int step=17+sizeof(Word)*dim,keysize=16+bits*step+sizeof(Word)*dim;
    for(int i=0;i<n;i++) {
        uint8_t *out[2]={out0+i*keysize,out1+i*keysize};Seed seeds[2];uint8_t controls[2]={0,1};
        Word along[2]={0,0};
        for(int b=0;b<2;b++){std::memcpy(seeds[b].v,roots+i*32+b*16,16);std::memcpy(out[b],seeds[b].v,16);}
        for(int level=0;level<bits;level++) {
            int keep=(alpha[i]>>(bits-1-level))&1,lose=keep^1;
            Seed child[2][2];uint8_t t[2][2];Word values[2][2][2];
            for(int b=0;b<2;b++)dcf_expand(seeds[b],child[b],t[b],values[b]);
            Word vcw[2];
            for(int c=0;c<dim;c++) {
                Word base=Word(Word(0-along[c])-values[0][lose][c]+values[1][lose][c]);
                if(keep)base=Word(base+Word(beta[i*dim+c]));
                vcw[c]=controls[1]?Word(0-base):base;
                Word signed_correction=controls[1]?Word(0-vcw[c]):vcw[c];
                along[c]=Word(along[c]-values[1][keep][c]+values[0][keep][c]+signed_correction);
            }
            Seed cw=sxor(child[0][lose],child[1][lose]);
            uint8_t tl=t[0][0]^t[1][0]^keep^1,tr=t[0][1]^t[1][1]^keep;
            for(int b=0;b<2;b++) {
                std::memcpy(out[b]+16+step*level,cw.v,16);out[b][16+step*level+16]=tl|(tr<<1);
                std::memcpy(out[b]+16+step*level+17,vcw,sizeof(Word)*dim);
                seeds[b]=child[b][keep];uint8_t nt=t[b][keep];
                if(controls[b]){seeds[b]=sxor(seeds[b],cw);nt^=keep?tr:tl;}controls[b]=nt;
            }
        }
        Word f0[2],f1[2];dcf_final(seeds[0],f0);dcf_final(seeds[1],f1);
        Word final[2];for(int c=0;c<dim;c++) {
            final[c]=Word(f1[c]-f0[c]-along[c]);if(controls[1])final[c]=Word(0-final[c]);
        }
        for(int b=0;b<2;b++)std::memcpy(out[b]+16+bits*step,final,sizeof(Word)*dim);
    }
    return 0;
}

template<class Word> int dcf_eval_impl(int bits,int dim,int n,int party,const uint8_t *keys,
    const uint64_t *x,uint64_t *outputs) {
    if(bits<1 || bits>63 || dim<1 || dim>2 || n<1 || party<0 || party>1)return -1;
    const int step=17+sizeof(Word)*dim,keysize=16+bits*step+sizeof(Word)*dim;
    for(int i=0;i<n;i++) {
        const uint8_t *key=keys+i*keysize;Seed seed;std::memcpy(seed.v,key,16);uint8_t control=party;Word sum[2]={0,0};
        for(int level=0;level<bits;level++) {
            int keep=(x[i]>>(bits-1-level))&1;Seed child[2];uint8_t t[2];Word values[2][2];
            dcf_expand(seed,child,t,values);Seed cw;std::memcpy(cw.v,key+16+step*level,16);
            uint8_t tcw=key[16+step*level+16];Word vcw[2]={0,0};std::memcpy(vcw,key+16+step*level+17,sizeof(Word)*dim);
            for(int c=0;c<dim;c++) {
                Word v=Word(values[keep][c]+(control?vcw[c]:Word(0)));
                sum[c]=Word(sum[c]+(party?Word(0-v):v));
            }
            seed=child[keep];uint8_t nt=t[keep];if(control){seed=sxor(seed,cw);nt^=(tcw>>keep)&1;}control=nt;
        }
        Word final[2]={0,0},v[2];std::memcpy(final,key+16+step*bits,sizeof(Word)*dim);dcf_final(seed,v);
        for(int c=0;c<dim;c++) {
            v[c]=Word(v[c]+(control?final[c]:Word(0)));
            outputs[i*dim+c]=uint64_t(Word(sum[c]+(party?Word(0-v[c]):v[c])));
        }
    }
    return 0;
}

extern "C" int dcf_generate_batch(int bits,int dim,int n,const uint64_t *alpha,const uint64_t *beta,
    const uint8_t *roots,uint8_t *out0,uint8_t *out1,int group_bits) {
    if(group_bits==16)return dcf_generate_impl<uint16_t>(bits,dim,n,alpha,beta,roots,out0,out1);
    if(group_bits==32)return dcf_generate_impl<uint32_t>(bits,dim,n,alpha,beta,roots,out0,out1);
    if(group_bits==64)return dcf_generate_impl<uint64_t>(bits,dim,n,alpha,beta,roots,out0,out1);
    return -1;
}
extern "C" int dcf_eval_batch(int bits,int dim,int n,int party,const uint8_t *keys,
    const uint64_t *x,uint64_t *outputs,int group_bits) {
    if(group_bits==16)return dcf_eval_impl<uint16_t>(bits,dim,n,party,keys,x,outputs);
    if(group_bits==32)return dcf_eval_impl<uint32_t>(bits,dim,n,party,keys,x,outputs);
    if(group_bits==64)return dcf_eval_impl<uint64_t>(bits,dim,n,party,keys,x,outputs);
    return -1;
}
