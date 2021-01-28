#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <iostream>
#include <string>
#include <chrono>
#include <iomanip>

using MemorySpace = Kokkos::DefaultExecutionSpace::memory_space;
using ExecutionSpace = Kokkos::DefaultExecutionSpace;
using DeviceType = Kokkos::Device<ExecutionSpace, MemorySpace>;
    
using kokkos_rng_pool_t = Kokkos::Random_XorShift64_Pool<ExecutionSpace>;
using kokkos_rng_state_t = Kokkos::Random_XorShift64<ExecutionSpace>;
    

KOKKOS_INLINE_FUNCTION
double uniform(const kokkos_rng_pool_t& r_p, const double low, const double high) {
    auto generator = r_p.get_state();
    auto r = generator.drand(low, high);
    r_p.free_state(generator);
    return r;
}

int main(int argc, char *argv[]) {

    //Initialize the kokkos runtime
    Kokkos::ScopeGuard scope_guard(argc, argv);
    
    printf("On Kokkos execution space %s\n", 
         typeid(Kokkos::DefaultExecutionSpace).name());
    
    std::size_t seed = 666;
    seed = std::chrono::system_clock::now().time_since_epoch().count();
    kokkos_rng_pool_t rand_pool(seed);

    using TestPolicy = Kokkos::TeamPolicy<ExecutionSpace>;
    using TestPolicyPar = Kokkos::RangePolicy<ExecutionSpace>;
    //320 cells seems to be where it maxes on GPU? (MOBILE GTX1060)

    const auto& _r_p = rand_pool;
    
    auto test_hier = KOKKOS_LAMBDA(TestPolicy::member_type team_member) {
        
        int cell = team_member.league_rank();
        double total = 0.0;
        for (int i = 0; i < 5000000; i++) {
            total += uniform(_r_p, 0.0, 1.0);
        }
        team_member.team_barrier();
        if(team_member.team_rank() == 0) {
            //printf("Job in cell %d has finished with total %f\n", cell, total);
        }
    };
    
    auto test_par = KOKKOS_LAMBDA(const int tid) {
        
        int cell = tid;
        double total = 0.0;
        for (int i = 0; i < 5000000; i++) {
            total += uniform(_r_p, 0.0, 1.0);
        }
    
        //printf("Job in cell %d has finished with total %f\n", cell, total);
        
    };    
    //TODO: prettify output
    //std::string headers[3] = {"Number of Cells", "Hier Time", "Par Time"};
    //std::cout << std::setw(headers[0].size()) << headers[0];
    
    {std::string title = "Hierarchical Tests";
    std::cout << title << std::endl;;
    for(auto i = 0; i < title.size(); i++) {std::cout << "-";}
    std::cout << std::endl;
    for(int i = 0; i <= 4; i++) {
        int num_cells = pow(2, i);
        TestPolicy test_policy(num_cells, 1);//Kokkos::AUTO);
        Kokkos::Timer timer;
        Kokkos::parallel_for("Test", test_policy, test_hier);
        Kokkos::fence();
        auto time = timer.seconds();
        std::cout << num_cells << " ran in "<< time << " seconds\n";
    }std::cout << "\n\n";}
    
    {std::string title = "Parallel For Tests";
    std::cout << title << std::endl;;
    for(auto i = 0; i < title.size(); i++) {std::cout << "-";}
    std::cout << std::endl;
     for(int i = 0; i <= 4; i++) {
        int num_cells = pow(2, i);
        TestPolicyPar test_policy(0, num_cells);//Kokkos::AUTO);
        Kokkos::Timer timer;
        Kokkos::parallel_for("Test", test_policy, test_par);
        Kokkos::fence();
        auto time = timer.seconds();
        std::cout << num_cells << " ran in "<< time << " seconds\n";
    }std::cout << "\n\n";}
 

    return 0;
}
