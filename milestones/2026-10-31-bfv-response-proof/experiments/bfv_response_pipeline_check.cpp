#include <seal/seal.h>
#include <seal/util/rns.h>
#include <chrono>
#include <fstream>
#include <iostream>
#include <set>
#include <string>
#include <vector>

// Link the frozen upstream function; no copy or modification of its sampler.
namespace utils {
void NoiseFloodInplace(seal::Ciphertext&, const seal::SEALContext&, size_t);
std::vector<long> get_perfect_constant_weight_codeword_position(uint64_t,uint64_t,uint64_t,bool);
}
void check(bool ok) { if (!ok) throw std::runtime_error("BFV pipeline correctness check"); }
std::uint32_t target(std::size_t c, std::size_t s) {
    switch (c) {
    case 0: return 0;
    case 1: return 1023;
    case 2: return s%1024;
    case 3: return s%4 ? 1024 : (s*37+17)%1024;
    case 4: return s%2 ? 255 : 512;
    default: return s%3 ? 1024 : (s*131+997)%1024;
    }
}
std::uint32_t data(std::size_t c, std::size_t col, std::size_t s) {
    constexpr auto mask = (1U<<17)-1;
    switch (c) {
    case 0: return (col*257+s*17+1)&mask;
    case 1: return mask;
    case 2: return (col+s)&1 ? mask : 0;
    case 3: return ((col+1)*(s+3)*13)&mask;
    case 4: return s%2 ? 1 : mask;
    default: return ((col*7)^(s*31)^mask)&mask;
    }
}
int main(int argc, char** argv) {
    if (argc != 4) throw std::runtime_error("case fixture.bin metrics.json");
    const auto c = std::stoul(argv[1]); check(c<6);
    auto start = std::chrono::steady_clock::now();
    seal::EncryptionParameters p(seal::scheme_type::bfv);
    p.set_poly_modulus_degree(8192);
    p.set_coeff_modulus(seal::CoeffModulus::Create(8192,{46,40,40,46,46}));
    p.set_plain_modulus(seal::PlainModulus::Batching(8192,18));
    seal::SEALContext context(p,true,seal::sec_level_type::tc128); check(context.parameters_set());
    seal::KeyGenerator keys(context); seal::RelinKeys rlk; keys.create_relin_keys(rlk);
    seal::PublicKey pk; keys.create_public_key(pk);
    seal::Encryptor symmetric(context,keys.secret_key()), asymmetric(context,pk);
    seal::Decryptor decryptor(context,keys.secret_key()); seal::Evaluator evaluator(context);
    seal::BatchEncoder encoder(context);
    constexpr std::size_t n=8192, columns=1024, m=46;
    std::vector<std::vector<long>> pairs(columns);
    std::set<std::vector<long>> seen;
    for (std::size_t col=0; col<columns; ++col) {
        pairs[col]=utils::get_perfect_constant_weight_codeword_position(col,m,2,false);
        check(pairs[col].size()==2 && seen.insert(pairs[col]).second);
    }
    std::vector<seal::Ciphertext> queries(m);
    std::vector<std::uint64_t> slots(n);
    for (std::size_t i=0; i<m; ++i) {
        for (std::size_t s=0; s<n; ++s) {
            auto t=target(c,s); slots[s]=t<columns && (pairs[t][0]==long(i) || pairs[t][1]==long(i));
        }
        seal::Plaintext pt; encoder.encode(slots,pt); symmetric.encrypt_symmetric(pt,queries[i]);
    }
    seal::Ciphertext response;
    for (std::size_t col=0; col<columns; ++col) {
        seal::Ciphertext mask;
        std::vector<seal::Ciphertext> selected{queries[pairs[col][0]],queries[pairs[col][1]]};
        evaluator.multiply_many(selected,rlk,mask); evaluator.transform_to_ntt_inplace(mask);
        for (std::size_t s=0; s<n; ++s) slots[s]=data(c,col,s);
        seal::Plaintext pt; encoder.encode(slots,pt);
        evaluator.transform_to_ntt_inplace(pt,context.first_parms_id());
        seal::Ciphertext value; evaluator.multiply_plain(mask,pt,value);
        if (col==0) response=std::move(value); else evaluator.add_inplace(response,value);
    }
    evaluator.transform_from_ntt_inplace(response);
    seal::Ciphertext zero; asymmetric.encrypt_zero(context.first_parms_id(),zero);
    evaluator.add_inplace(response,zero);
    utils::NoiseFloodInplace(response,context,149);
    evaluator.mod_switch_to_inplace(response,context.last_parms_id());
    seal::Plaintext answer; decryptor.decrypt(response,answer); encoder.decode(answer,slots);
    std::ofstream fixture(argv[2],std::ios::binary); check(bool(fixture));
    for (std::size_t s=0; s<n; ++s) {
        auto t=target(c,s), expected=t<columns ? data(c,t,s) : 0;
        check(slots[s]==expected);
        std::uint32_t words[3]{t,expected,static_cast<std::uint32_t>(slots[s])};
        fixture.write(reinterpret_cast<char*>(words),sizeof(words));
    }
    fixture.close(); check(bool(fixture));
    auto rns=context.first_context_data()->rns_tool();
    auto elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    std::ofstream out(argv[3]); check(bool(out));
    out << "{\"all_passed\":true,\"case\":" << c << ",\"columns\":1024,\"slots_checked\":8192,"
        << "\"m\":46,\"noise_bits\":149,\"fresh_publickey_zero\":true,\"base_B\":[";
    for (std::size_t i=0;i<rns->base_B()->size();++i) {
        if (i) out << ",";
        out << (*rns->base_B())[i].value();
    }
    out << "],\"m_sk\":" << rns->m_sk().value() << ",\"m_tilde\":" << rns->m_tilde().value()
        << ",\"elapsed_seconds\":" << elapsed << "}\n";
    check(bool(out));
    std::cout << "case " << c << " passed8192slots, seconds=" << elapsed << std::endl;
}
