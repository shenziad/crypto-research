#pragma once
#include "public_tanh_ot.h"
#include <iostream>

namespace ot_pwl_chunked {
inline std::vector<uint64_t> evaluate(LinearOT& linear,std::vector<uint64_t>& raw_z) {
    // Public batching of a coordinatewise function; neither values nor
    // acceptance events change the fixed schedule. Same OTPack advances state.
    constexpr size_t chunk=4096;
    std::vector<uint64_t> result;result.reserve(raw_z.size());
    for(size_t start=0;start<raw_z.size();start+=chunk) {
        size_t end=std::min(start+chunk,raw_z.size());
        std::vector<uint64_t> input(raw_z.begin()+start,raw_z.begin()+end);
        std::cout<<"PUBLIC_OT_PWL_CHUNK begin="<<start<<" count="<<end-start<<std::endl;
        auto output=ot_pwl::evaluate(linear,input);
        result.insert(result.end(),output.begin(),output.end());
        std::cout<<"PUBLIC_OT_PWL_CHUNK complete="<<end<<std::endl;
    }
    return result;
}
} // namespace ot_pwl_chunked
