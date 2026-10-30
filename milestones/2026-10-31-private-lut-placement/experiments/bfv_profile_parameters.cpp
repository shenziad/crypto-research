#include <seal/seal.h>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

void moduli(const seal::EncryptionParameters& p) {
    std::cout << "[";
    for (std::size_t i = 0; i < p.coeff_modulus().size(); ++i) {
        if (i) std::cout << ",";
        std::cout << p.coeff_modulus()[i].value();
    }
    std::cout << "]";
}
int main() {
    struct Spec { const char* name; int plain; std::vector<int> coeff; int flood; };
    std::vector<Spec> specs = {
        {"measured_qwen18",20,{45,45,50},69},
        {"upstream_case20_reference",18,{46,40,40,46,46},149}
    };
    std::cout << "[";
    bool first = true;
    for (const auto& spec : specs) {
        seal::EncryptionParameters p(seal::scheme_type::bfv);
        p.set_poly_modulus_degree(8192);
        p.set_coeff_modulus(seal::CoeffModulus::Create(8192,spec.coeff));
        p.set_plain_modulus(seal::PlainModulus::Batching(8192,spec.plain));
        seal::SEALContext ctx(p, true, seal::sec_level_type::tc128);
        if (!ctx.parameters_set()) throw std::runtime_error(ctx.parameter_error_message());
        if (!first) std::cout << ",";
        first = false;
        std::cout << "{\"name\":\"" << spec.name << "\",\"n\":8192,\"plain_bits\":" << spec.plain
            << ",\"plain_modulus\":" << p.plain_modulus().value()
            << ",\"noise_bits\":" << spec.flood << ",\"key_moduli\":";
        moduli(p);
        std::cout << ",\"first_moduli\":"; moduli(ctx.first_context_data()->parms());
        std::cout << ",\"last_moduli\":"; moduli(ctx.last_context_data()->parms());
        std::cout << ",\"SEAL_tc128_parameters_accepted\":true}";
    }
    std::cout << "]\n";
}
