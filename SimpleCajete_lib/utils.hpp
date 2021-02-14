#ifndef __CAJETE_UTILS_HPP
#define __CAJETE_UTILS_HPP

#include <Kokkos_Random.hpp>
#include <chrono>

using kokkos_rng_pool_h_t = Kokkos::Random_XorShift64_Pool<Kokkos::DefaultHostExecutionSpace>;
using kokkos_rng_state_h_t = Kokkos::Random_XorShift64<Kokkos::DefaultHostExecutionSpace>;


//perform a fisher yates shuffle on a 1D view
template<typename ViewType>
void fisher_yates(ViewType& permute) {
    //set up random number generator
    auto seed = std::chrono::system_clock::now().time_since_epoch().count();
    kokkos_rng_pool_h_t rand_pool(seed);

    auto size = permute.size();
    
    //at start of kernel
    auto rand_gen = rand_pool.get_state();
    
    for(auto i = size-1; i > 1; i--) { 
        int j = rand_gen.rand(0, i);
        int temp = permute(i);
        permute(i) = permute(j);
        permute(j) = temp;
    }
    
    rand_pool.free_state(rand_gen);

}

template<typename ViewType>
void permute_fill(ViewType& permute) {
    auto size = permute.size();

    Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace> fill_policy(0, size);
    Kokkos::parallel_for("PermuteFill", fill_policy, KOKKOS_LAMBDA(const int i) {    
        permute(i) = i;
    });
    Kokkos::fence();
}

#endif
