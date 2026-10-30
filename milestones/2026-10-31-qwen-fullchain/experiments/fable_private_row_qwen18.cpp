// Harness for the pinned, unmodified FABLE lookup algorithm. Adds private
// XOR-share inputs and masked XOR-share outputs around its published API.
// No intermediate index or row is revealed. Local files are party-specific.
#include "GC/emp-sh2pc.h"
#include "GC/lookup.h"
#include "database_constants.h"
#include <array>
#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <sys/random.h>

using namespace sci;
using Clock = std::chrono::steady_clock;

static std::vector<unsigned char> load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open party input");
    return std::vector<unsigned char>(std::istreambuf_iterator<char>(f), {});
}

static void random_bytes(unsigned char* p, size_t n) {
    size_t at=0;
    while (at<n) {
        auto got=getrandom(p+at,n-at,0);
        if (got<=0) throw std::runtime_error("getrandom failed");
        at+=got;
    }
}

static double ms(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double,std::milli>(b-a).count();
}

int main(int argc, char** argv) {
    try {
        if(argc!=9) throw std::runtime_error("role port count table-or-dash indexshare outputshare metrics threads");
        int party=std::stoi(argv[1]),port=std::stoi(argv[2]);
        size_t count=std::stoul(argv[3]);
        int threads=std::stoi(argv[8]);
        if(party!=ALICE && party!=BOB) throw std::runtime_error("bad role");
        if(LUT_INPUT_SIZE!=18 || LUT_OUTPUT_SIZE!=256 || count==0 || (count&(count-1)))
            throw std::runtime_error("requires fixed18x256 and power2 batch");
        constexpr size_t n=262144,width=32;
        auto shares=load(argv[5]);
        if(shares.size()!=4*count) throw std::runtime_error("bad share count");
        std::map<uint64_t,utils::rawdatablock> table;
        if(party==ALICE) {
            auto raw=load(argv[4]);
            if(raw.size()!=n*width) throw std::runtime_error("bad row table size");
            for(size_t i=0;i<n;i++) {
                utils::rawdatablock row;
                for(size_t j=0;j<width*8;j++) row[j]=(raw[i*width+j/8]>>(j%8))&1;
                table.emplace(i,row);
            }
        } else if(std::string(argv[4])!="-") throw std::runtime_error("nonowner must have no table path");
        auto t0=Clock::now();
        NetIO io(party==ALICE?nullptr:"127.0.0.1",port,true);
        auto connected=Clock::now();
        setup_semi_honest(&io,party);
        auto gc_ready=Clock::now();
        auto gc_bytes=io.counter, gc_recv=io.received_counter;
        auto params=fable_prepare(table,party,count,n,true,threads,
                                  BatchPirType::PIRANA,HashType::AES,&io);
        io.flush();
        auto prepared=Clock::now();
        auto prepare_bytes=io.counter,prepare_recv=io.received_counter;
        IntegerArray queries;
        for(size_t i=0;i<count;i++) {
            uint32_t value=0;for(int j=0;j<4;j++)value|=uint32_t(shares[4*i+j])<<(8*j);
            if(value>=(1u<<18))throw std::runtime_error("bad18bitindexshare");
            Integer left(DatabaseConstants::InputLength+1,party==ALICE?value:0,ALICE);
            Integer right(DatabaseConstants::InputLength+1,party==BOB?value:0,BOB);
            queries.push_back(left^right);
        }
        barrier(party,&io);io.flush();
        auto inputs_ready=Clock::now();
        auto input_bytes=io.counter,input_recv=io.received_counter;
        auto result=fable_lookup(queries,params,true);
        io.flush();
        auto looked_up=Clock::now();
        auto lookup_bytes=io.counter,lookup_recv=io.received_counter;
        if(result.size()!=count) throw std::runtime_error("bad result length");
        std::vector<unsigned char> output(count*width,0);
        for(size_t i=0;i<count;i++) {
            utils::rawdatablock pad;
            if(party==ALICE) {
                random_bytes(output.data()+i*width,width);
                for(size_t j=0;j<width*8;j++) pad[j]=(output[i*width+j/8]>>(j%8))&1;
            }
            auto masked=result[i]^share_bitset(pad,ALICE);
            for(size_t j=0;j<width*8;j++) {
                bool bit=masked[j].reveal<bool>(BOB);
                if(party==BOB && bit) output[i*width+j/8] |= uint8_t(1u<<(j%8));
            }
        }
        io.flush();
        auto done=Clock::now();
        std::ofstream out(argv[6],std::ios::binary);
        out.write(reinterpret_cast<char*>(output.data()),output.size());out.close();
        if(!out) throw std::runtime_error("share write failed");
        std::ofstream metric(argv[7]);
        metric << "{\"role\":" << party << ",\"queries\":" << count
               << ",\"row_bytes\":32,\"oprf\":\"AES128\",\"threads\":" << threads
               << ",\"connect_wait_ms\":" << ms(t0,connected)
               << ",\"gc_setup_ms\":" << ms(connected,gc_ready)
               << ",\"prepare_ms\":" << ms(gc_ready,prepared)
               << ",\"input_ms\":" << ms(prepared,inputs_ready)
               << ",\"lookup_ms\":" << ms(inputs_ready,looked_up)
               << ",\"share_release_ms\":" << ms(looked_up,done)
               << ",\"protocol_ms\":" << ms(connected,done)
               << ",\"sent\":" << io.counter << ",\"received\":" << io.received_counter
               << ",\"gc_setup_sent\":" << gc_bytes << ",\"gc_setup_received\":" << gc_recv
               << ",\"prepare_sent\":" << prepare_bytes-gc_bytes
               << ",\"prepare_received\":" << prepare_recv-gc_recv
               << ",\"input_sent\":" << input_bytes-prepare_bytes
               << ",\"input_received\":" << input_recv-prepare_recv
               << ",\"lookup_sent\":" << lookup_bytes-input_bytes
               << ",\"lookup_received\":" << lookup_recv-input_recv
               << ",\"release_sent\":" << io.counter-lookup_bytes
               << ",\"release_received\":" << io.received_counter-lookup_recv
               << ",\"io_direction_changes\":" << io.num_rounds << "}\n";
        std::cout << "PRIVATE_ROW_LOOKUP_PROTOCOL_FINISHED role=" << party << std::endl;
        return 0;
    } catch(const std::exception& e) {
        std::cerr << "harness error: " << e.what() << std::endl;
        return 1;
    }
}
