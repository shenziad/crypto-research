#include "GC/emp-sh2pc.h"
#include "GC/fable_dual_domain_aes.h"
#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>

using namespace sci;
static std::vector<unsigned char> load(const char* name) {
    std::ifstream f(name,std::ios::binary);
    if (!f) throw std::runtime_error("AEScontrolinput");
    return {std::istreambuf_iterator<char>(f),{}};
}
int main(int argc,char** argv) {
    try {
        if (argc!=7) throw std::runtime_error("role port owninput output metrics count");
        int role=std::stoi(argv[1]),port=std::stoi(argv[2]);std::size_t count=std::stoul(argv[6]);
        if (role!=ALICE && role!=BOB) throw std::runtime_error("role");
        auto own=load(argv[3]);if (own.size()!=16+4*count) throw std::runtime_error("owninputsize");
        alignas(16) unsigned char keybytes[16]{};
        if (role==ALICE) std::copy_n(own.data(),16,keybytes);
        osuCrypto::block key(_mm_load_si128(reinterpret_cast<const __m128i*>(keybytes)));
        auto start=std::chrono::steady_clock::now();
        NetIO io(role==ALICE?nullptr:"127.0.0.1",port,true);setup_semi_honest(&io,role);
        IntegerArray queries;
        for (std::size_t i=0;i<count;++i) {
            std::uint32_t value=0;
            for (unsigned j=0;j<4;++j)value|=std::uint32_t(own[16+4*i+j])<<(8*j);
            if (value>=(1u<<19)) throw std::runtime_error("sharewidth");
            Integer a(19,role==ALICE?value:0,ALICE),b(19,role==BOB?value:0,BOB);
            queries.push_back(a^b);
        }
        auto tags=proof_dual_domain_aes(queries,key);io.flush();
        std::ofstream out(argv[4],std::ios::binary);
        if (role==BOB) for (const auto& tag:tags) for (unsigned domain=0;domain<2;++domain) {
            std::bitset<128> bits(tag.substr(domain*128,128));
            for (unsigned byte=0;byte<16;++byte) {
                unsigned char value=0;
                for (unsigned bit=0;bit<8;++bit)if(bits[byte*8+bit])value|=1u<<bit;
                out.put(static_cast<char>(value));
            }
        }
        out.close();if(!out)throw std::runtime_error("write");
        auto elapsed=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
        std::ofstream metric(argv[5]);metric << "{\"role\":" << role << ",\"count\":" << count
            << ",\"domains\":2,\"sent\":" << io.counter << ",\"received\":" << io.received_counter
            << ",\"elapsed_ms\":" << elapsed << "}\n";
        return 0;
    } catch(const std::exception& e) {std::cerr << e.what() << std::endl;return 1;}
}
