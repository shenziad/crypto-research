// Typed extension for full finite-block GPU execution. Old library untouched.
#include "dcf_cuda.cu"

__device__ __forceinline__ U le32(const unsigned char *p){return U(p[0])|(U(p[1])<<8)|(U(p[2])<<16)|(U(p[3])<<24);}
__global__ void select32_kernel(int bits,int dim,int n,int queries,int party,int count,int per_query,
    const unsigned char *keys,const V *x,const int *coords,V *out){
    int index=blockIdx.x*blockDim.x+threadIdx.x;if(index>=n*queries)return;
    int row=index/queries,step=17+4*dim,size=16+bits*step+4*dim;
    const unsigned char *key=keys+size_t(row)*size;const int *co=coords+(per_query?size_t(index)*count:0);
    State seed=load(key);unsigned control=party;U sums[2]={0};V xx=x[index];
    for(int level=0;level<bits;level++){
        U rk[44];schedule(seed,rk);int branch=(xx>>(bits-1-level))&1;
        State next=counter(rk,branch),tc=counter(rk,2),value=counter(rk,3+branch);
        for(int j=0;j<count;j++){
            int c=co[j];U v=sw(value.v[c])+(control?le32(key+16+step*level+17+4*c):0);sums[j]+=party?U(0-v):v;
        }
        unsigned nt=(sw(tc.v[0])>>branch)&1;
        if(control){State cw=load(key+16+step*level);
            #pragma unroll
            for(int j=0;j<4;j++)next.v[j]^=cw.v[j];
            nt^=(key[16+step*level+16]>>branch)&1;
        }
        seed=next;control=nt;
    }
    U rk[44];schedule(seed,rk);State value=counter(rk,5);
    for(int j=0;j<count;j++){
        int c=co[j];U v=sw(value.v[c])+(control?le32(key+16+step*bits+4*c):0);
        out[size_t(index)*count+j]=U(sums[j]+(party?U(0-v):v));
    }
}
extern "C" int dcf_cuda_select_typed(int bits,int dim,int n,int queries,int party,int count,int per_query,int group_bits,
    const unsigned char *keys,const V *x,const int *coords,V *out,void *stream){
    if(group_bits==64)return dcf_cuda_select(bits,dim,n,queries,party,count,per_query,keys,x,coords,out,stream);
    if(group_bits!=32||dim<1||dim>2||count<1||count>2||bits<1||bits>63||n<1||queries<1||party<0||party>1)return -1;
    select32_kernel<<<(n*queries+127)/128,128,0,(cudaStream_t)stream>>>(bits,dim,n,queries,party,count,per_query,keys,x,coords,out);
    return int(cudaGetLastError());
}
__global__ void contraction_kernel(int n,const V *public_w,const V *mask_w,const V *mask_v,const V *zhat,const V *product,int party,V *out){
    __shared__ V partial[128];int row=blockIdx.x,t=threadIdx.x;V sum=0;
    for(int j=t;j<n;j+=128){V d=public_w[size_t(row)*n+j],a=mask_w[size_t(row)*n+j],z=zhat[j];sum-=d*mask_v[j]+a*z;if(!party)sum+=d*z;}
    partial[t]=sum;__syncthreads();
    for(int d=64;d>0;d/=2){if(t<d)partial[t]+=partial[t+d];__syncthreads();}
    if(t==0)out[row]=(partial[0]+product[row])&0x7fffffffffffffffULL;
}
extern "C" int cuda_private_contraction(int m,int n,const V *public_w,const V *mask_w,const V *mask_v,const V *zhat,const V *product,int party,V *out,void *stream){
    contraction_kernel<<<m,128,0,(cudaStream_t)stream>>>(n,public_w,mask_w,mask_v,zhat,product,party,out);return int(cudaGetLastError());
}
