#pragma once
#include "aes.h"
#include "custom_types.h"
#include <cryptoTools/Common/block.h>
#include <bitset>
#include <stdexcept>
#include <string>
#include <vector>

namespace sci {
// Actual protocol path and the independent two-party AES control share this
// function. Inputs are19-bit valid IDs; domain occupies bit127 of LE128 input.
inline std::vector<std::string> proof_dual_domain_aes(
        const IntegerArray& queries, const osuCrypto::block& own_key) {
    const auto words=own_key.get<std::uint64_t>();
    Integer key(128,0);
    for (int j=0;j<128;++j) key[j]=Bit((words[j/64]>>(j%64))&1,ALICE);
    AES cipher(key);
    std::vector<std::string> tags(queries.size());
    for (int domain=0;domain<2;++domain) {
        std::vector<Integer> inputs;
        for (const auto& query:queries) {
            if (query.size()!=19) throw std::invalid_argument("dual-domain IDwidth");
            Integer input(128,0);
            for (int j=0;j<19;++j) input[j]=query[j];
            input[127]=Bit(domain);
            inputs.push_back(std::move(input));
        }
        auto encrypted=cipher.EncryptECB(inputs);
        for (std::size_t i=0;i<queries.size();++i) {
            std::bitset<128> bits;
            for (int j=0;j<128;++j) bits[j]=encrypted[i][j].reveal(BOB);
            tags[i]+=bits.to_string();
        }
    }
    return tags;
}
} // namespace sci
