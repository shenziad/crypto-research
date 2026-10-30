// Read the actual pinned SCI MathFunctions public factors; no private inputs.
#include <cstdint>
#include <iostream>
uint64_t lookup_neg_exp(uint64_t val_in,int32_t s_in,int32_t s_out);
int main() {
    std::cout<<"{\"input_bits\":22,\"input_fraction\":20,\"output_fraction\":31,\"digit_bits\":8,\"tables\":[";
    for(int digit=0;digit<3;digit++) {
        if(digit)std::cout<<",";std::cout<<"[";
        int count=digit<2?256:64;
        for(int j=0;j<count;j++) {if(j)std::cout<<",";std::cout<<lookup_neg_exp(j,20-8*digit,31);}
        std::cout<<"]";
    }
    std::cout<<"]}"<<std::endl;return 0;
}
