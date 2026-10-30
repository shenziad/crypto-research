#include "private_weighted_cdf93.h"
#include "utils/io_pack.h"
#include "OT/ot_pack.h"
#include <chrono>
#include <fstream>
#include <iostream>

static uint64_t read64(std::ifstream& f){uint64_t v=0;for(int j=0;j<8;j++){int c=f.get();if(c==EOF)throw std::runtime_error("shortinput");v|=uint64_t(uint8_t(c))<<(8*j);}return v;}
int main(int argc,char** argv){
    try{
        if(argc!=6)throw std::runtime_error("role port rows input ownfolder");
        int role=std::stoi(argv[1]),port=std::stoi(argv[2]),n=std::stoi(argv[3]);
        if(n<1 || n>50304 || (role!=1 && role!=2))throw std::runtime_error("badshape");
        std::ifstream f(argv[4],std::ios::binary);if(!f)throw std::runtime_error("noinput");
        std::vector<uint64_t> e(n),q(n);for(int i=0;i<n;i++){e[i]=read64(f);q[i]=read64(f);if(e[i]>>33)throw std::runtime_error("badexpwidth");}
        uint64_t u=read64(f);if(f.get()!=EOF)throw std::runtime_error("trailinginput");
        auto start=std::chrono::steady_clock::now();sci::IOPack io(role,port,"127.0.0.1");sci::OTPack ot(&io,role,true);LinearOT linear(role,&io,&ot);
        private_cdf93::sample(linear,io,e,q,u,argv[5],false);
        io.io->flush();io.io_rev->flush();io.io_GC->flush();auto done=std::chrono::steady_clock::now();
        std::ofstream m(std::string(argv[5])+"/metrics.json");m<<"{\"role\":"<<role<<",\"rows\":"<<n
          <<",\"protocol_ms\":"<<std::chrono::duration<double,std::milli>(done-start).count()
          <<",\"sent\":"<<io.io->counter+io.io_rev->counter+io.io_GC->counter
          <<",\"received\":"<<io.io->received_counter+io.io_rev->received_counter+io.io_GC->received_counter<<"}\n";
        std::cout<<"WEIGHTED_CDF_FINISHED role="<<role<<std::endl;return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}
}
