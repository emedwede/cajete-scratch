#ifndef __CAJETE_MT_INITS_HPP
#define __CAJETE_MT_INITS_HPP

#include <Kokkos_Random.hpp>
#include "graph.hpp"

#include <chrono>
using kokkos_rng_pool_t = Kokkos::Random_XorShift64_Pool<Kokkos::DefaultExecutionSpace>;
using kokkos_rng_state_t = Kokkos::Random_XorShift64<Kokkos::DefaultExecutionSpace>;

/*****************************************************
* Takes number of microtubules to uniformly distribute
*
* In this case we asuume the simple setup:
*
* negative <--> intermediate <--> positive
*
* Hence the name mt3 == microtuble 3 nodes
******************************************************/
//TODO: Determine if it is more wise to pass system by value or reference
//      Seems debatable since object has low official memory overhead thanks
//      to how kokkos uses shared pointers with reference counts
template<typename GraphType>
void mt3_uniform_init(GraphType& system, size_t nmt) {
    //we can only fill in multiples of 3, so check system stats
    //and resize if needed
    if(system.get_size() % 3 != 0) {
        system.resize_HostSafe(3*nmt);    
    }

    //set up random number generator
    auto seed = std::chrono::system_clock::now().time_since_epoch().count();
    kokkos_rng_pool_t rand_pool(seed);
    
    //make gpu capturable
    const auto& _r_p = rand_pool;
    //This capute convention is only needed when be pass by reference not value
    //const auto& _system = system;
    Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace> fill_policy(0, nmt);
    
    Kokkos::parallel_for("MT::Init", fill_policy, KOKKOS_LAMBDA(const int i) {    
        //at start of kernel
        auto rand_gen = _r_p.get_state();
        
        double c_x, c_y, x1, x3, y1, y3, theta, radius;

        //hard coded for now, but later should rely on cell complex
        //domain boundary corners
        double lx = 0.0, ly = 0.0, ux = 10.0, uy = 10.0;
        double LENGTH_MAX = 0.5; //Max segemnt lenght
        
        //pad to wind up only on interior
        c_x = rand_gen.drand(lx+LENGTH_MAX, ux-LENGTH_MAX);
        c_y = rand_gen.drand(ly+LENGTH_MAX, uy-LENGTH_MAX);
        theta = rand_gen.drand(0.0, 2.0*3.1459265);
        
        //TODO: remove the global constant
        radius = rand_gen.drand(0.5*LENGTH_MAX, 0.8*LENGTH_MAX);
        
        x1 = radius*cos(theta)+c_x;
        y1 = radius*sin(theta)+c_y;
        x3 = c_x-radius*cos(theta);
        y3 = c_y-radius*sin(theta);
        
        int ppmt = 3;
        int k = i*ppmt;
        //set the negative end
        system.positions_d(k, 0) = x1;
        system.positions_d(k, 1) = y1;
        system.nodes_d(k) = 0;
        system.edges_d(k, 0) = k+1;
        system.id_d(k) = k;

        //set the intermediate
        system.positions_d(k+1, 0) = c_x;
        system.positions_d(k+1, 1) = c_y;
        system.nodes_d(k+1) = 1;
        system.edges_d(k+1, 0) = k;
        system.edges_d(k+1, 1) = k+2;
        system.id_d(k+1) = k+1;

        system.positions_d(k+2, 0) = x3;
        system.positions_d(k+2, 1) = y3;
        system.nodes_d(k+2) = 2;
        system.edges_d(k+2, 0) = k+1;
        system.id_d(k+2) = k+2;
        //at end of the kernel
        _r_p.free_state(rand_gen);
    });
    
    system.copy_device_to_host();
}

#endif
