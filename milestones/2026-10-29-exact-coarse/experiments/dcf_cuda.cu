// Research CUDA evaluator for the exact existing AES seed-as-key DCF wire format.
// Own AES implementation follows FIPS197; no upstream GPU speed/security claim.
#include <cuda_runtime.h>
#include <stdint.h>

using U=uint32_t;using V=uint64_t;
__device__ U table[256];
__device__ unsigned char sbox[256];

__device__ __forceinline__ U rr(U x,int n){return (x>>n)|(x<<(32-n));}
__device__ __forceinline__ U sw(U x){return __byte_perm(x,0,0x0123);}
__device__ __forceinline__ U be(const unsigned char *p){return (U(p[0])<<24)|(U(p[1])<<16)|(U(p[2])<<8)|p[3];}
__device__ __forceinline__ V le64(const unsigned char *p){V v=0;
    #pragma unroll
    for(int j=0;j<8;j++)v|=V(p[j])<<(8*j);return v;}
__device__ __forceinline__ U sub(U x){return (U(__ldg(sbox+(x>>24)))<<24)|(U(__ldg(sbox+((x>>16)&255)))<<16)|(U(__ldg(sbox+((x>>8)&255)))<<8)|__ldg(sbox+(x&255));}
struct State{U v[4];};
__device__ __forceinline__ State load(const unsigned char *p){State a;
    #pragma unroll
    for(int j=0;j<4;j++)a.v[j]=be(p+4*j);return a;}
__device__ __forceinline__ void schedule(State a,U *rk){
    #pragma unroll
    for(int j=0;j<4;j++)rk[j]=a.v[j];
    U rc=1;
    #pragma unroll
    for(int r=1;r<=10;r++){
        U t=sub((a.v[3]<<8)|(a.v[3]>>24))^(rc<<24);
        a.v[0]^=t;a.v[1]^=a.v[0];a.v[2]^=a.v[1];a.v[3]^=a.v[2];
        #pragma unroll
        for(int j=0;j<4;j++)rk[4*r+j]=a.v[j];
        rc=((rc<<1)^((rc&128)?0x11b:0))&255;
    }
}
__device__ __forceinline__ State encrypt(const U *rk,State a){
    #pragma unroll
    for(int j=0;j<4;j++)a.v[j]^=rk[j];
    #pragma unroll
    for(int r=1;r<10;r++){
        State b;
        #pragma unroll
        for(int j=0;j<4;j++)b.v[j]=__ldg(table+(a.v[j]>>24))^rr(__ldg(table+((a.v[(j+1)%4]>>16)&255)),8)^rr(__ldg(table+((a.v[(j+2)%4]>>8)&255)),16)^rr(__ldg(table+(a.v[(j+3)%4]&255)),24)^rk[4*r+j];
        a=b;
    }
    State b;
    #pragma unroll
    for(int j=0;j<4;j++)b.v[j]=(U(__ldg(sbox+(a.v[j]>>24)))<<24)|(U(__ldg(sbox+((a.v[(j+1)%4]>>16)&255)))<<16)|(U(__ldg(sbox+((a.v[(j+2)%4]>>8)&255)))<<8)|__ldg(sbox+(a.v[(j+3)%4]&255));
    #pragma unroll
    for(int j=0;j<4;j++)b.v[j]^=rk[40+j];return b;
}
__device__ __forceinline__ State counter(const U *rk,V ctr){State a={{sw(U(ctr)),sw(U(ctr>>32)),0x31764643,0x444d4c4c}};return encrypt(rk,a);}
__device__ __forceinline__ V word(State a,int parity){return V(sw(a.v[2*parity]))|(V(sw(a.v[2*parity+1]))<<32);}

__global__ void aes_kernel(int n,const unsigned char *keys,const unsigned char *plain,unsigned char *out){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;U rk[44];schedule(load(keys+16*i),rk);State a=encrypt(rk,load(plain+16*i));
    #pragma unroll
    for(int j=0;j<4;j++){
        #pragma unroll
        for(int k=0;k<4;k++)out[16*i+4*j+k]=(a.v[j]>>(24-8*k))&255;
    }
}

__global__ void select_kernel(int bits,int dim,int n,int queries,int party,int count,int per_query,
    const unsigned char *keys,const V *x,const int *coords,V *out){
    int index=blockIdx.x*blockDim.x+threadIdx.x;if(index>=n*queries)return;
    int row=index/queries,step=17+8*dim,size=16+bits*step+8*dim,blocks=(dim+1)/2;
    const unsigned char *key=keys+size_t(row)*size;const int *co=coords+(per_query?size_t(index)*count:0);
    State seed=load(key);unsigned control=party;V sums[8]={0};V xx=x[index];
    for(int level=0;level<bits;level++){
        U rk[44];schedule(seed,rk);int branch=(xx>>(bits-1-level))&1;
        State next=counter(rk,branch),tc=counter(rk,2),value;int previous=-1;
        for(int j=0;j<count;j++){
            int c=co[j],block=c/2;if(block!=previous){value=counter(rk,3+2*block+branch);previous=block;}
            V v=word(value,c&1)+(control?le64(key+16+step*level+17+8*c):0);sums[j]+=party?V(0-v):v;
        }
        unsigned nt=(sw(tc.v[0])>>branch)&1;
        if(control){State cw=load(key+16+step*level);
            #pragma unroll
            for(int j=0;j<4;j++)next.v[j]^=cw.v[j];
            nt^=(key[16+step*level+16]>>branch)&1;
        }
        seed=next;control=nt;
    }
    U rk[44];schedule(seed,rk);State value;int previous=-1;
    for(int j=0;j<count;j++){
        int c=co[j],block=c/2;if(block!=previous){value=counter(rk,3+2*blocks+block);previous=block;}
        V v=word(value,c&1)+(control?le64(key+16+step*bits+8*c):0);out[size_t(index)*count+j]=sums[j]+(party?V(0-v):v);
    }
}

extern "C" int dcf_cuda_init(const unsigned char *host_sbox,const U *host_table){
    auto a=cudaMemcpyToSymbol(sbox,host_sbox,256);if(a!=cudaSuccess)return int(a);
    return int(cudaMemcpyToSymbol(table,host_table,256*sizeof(U)));
}
extern "C" int aes_cuda_eval(int n,const unsigned char *keys,const unsigned char *plain,unsigned char *out,void *stream){
    aes_kernel<<<(n+127)/128,128,0,(cudaStream_t)stream>>>(n,keys,plain,out);return int(cudaGetLastError());
}
extern "C" int dcf_cuda_select(int bits,int dim,int n,int queries,int party,int count,int per_query,
    const unsigned char *keys,const V *x,const int *coords,V *out,void *stream){
    if(bits<1||bits>63||dim<1||dim>128||n<1||queries<1||count<1||count>8||party<0||party>1)return -1;
    select_kernel<<<(n*queries+127)/128,128,0,(cudaStream_t)stream>>>(bits,dim,n,queries,party,count,per_query,keys,x,coords,out);
    return int(cudaGetLastError());
}
