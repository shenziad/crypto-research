// Actual two-party exhaustive finite interpolation-core validation.
#include "public_tanh_ot_chunked.h"
#include "utils/io_pack.h"
#include "OT/ot_pack.h"
#include <chrono>
#include <fstream>
#include <iostream>

using namespace sci;
using Clock=std::chrono::steady_clock;
int main(int argc,char** argv) {
    try {
        if(argc!=6)throw std::runtime_error("role port input output metrics");
        int role=std::stoi(argv[1]),port=std::stoi(argv[2]);
        if(role!=ALICE && role!=BOB)throw std::runtime_error("badrole");
        std::ifstream f(argv[3],std::ios::binary);if(!f)throw std::runtime_error("noinput");
        std::vector<uint64_t> input;
        while(f.peek()!=EOF) {
            uint64_t v=0;
            for(int j=0;j<8;j++){int c=f.get();if(c==EOF)throw std::runtime_error("shortword");v|=uint64_t(uint8_t(c))<<(8*j);}
            if(v>>44)throw std::runtime_error("badwidth");input.push_back(v);
        }
        auto start=Clock::now();IOPack io(role,port,"127.0.0.1");OTPack ot(&io,role,true);LinearOT linear(role,&io,&ot);
        auto output=ot_pwl_chunked::evaluate(linear,input);
        io.io->flush();io.io_rev->flush();io.io_GC->flush();auto done=Clock::now();
        std::ofstream out(argv[4],std::ios::binary);
        for(auto v:output)for(int j=0;j<8;j++)out.put(char((v>>(8*j))&255));out.close();
        std::ofstream m(argv[5]);
        m<<"{\"role\":"<<role<<",\"values\":"<<input.size()<<",\"protocol_ms\":"
         <<std::chrono::duration<double,std::milli>(done-start).count()
         <<",\"sent\":"<<io.io->counter+io.io_rev->counter+io.io_GC->counter
         <<",\"received\":"<<io.io->received_counter+io.io_rev->received_counter+io.io_GC->received_counter<<"}\n";
        return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}
}
