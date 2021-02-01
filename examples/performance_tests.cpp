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

KOKKOS_INLINE_FUNCTION
void compute_test(typename kokkos_rng_pool_t::generator_type& gen, const Kokkos::View<size_t*>& garbage, const size_t size, const size_t div) {
    for(size_t i = 0; i < size; i++) {
        for(size_t j = 0; j < size/div; j++) {
            garbage(i) += i*j;
            garbage(i) += gen.drand(0.0, 100.0);
        }
    }
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
    size_t size = 1000000;
    size_t div = 30000;
    Kokkos::View<size_t*> garbage("garbage", size);
    Kokkos::parallel_for("Test init", size, KOKKOS_LAMBDA(const size_t i) {
        garbage(i) = i;
    });
    auto test_hier = KOKKOS_LAMBDA(TestPolicy::member_type team_member) {
          
        int cell = team_member.league_rank();
        
        Kokkos::Random_XorShift64_Pool<ExecutionSpace>::generator_type rand_gen = _r_p.get_state(); 
        compute_test(rand_gen, garbage, size, div);
        _r_p.free_state(rand_gen);
    };
   
        auto test_par = KOKKOS_LAMBDA(const int tid) {
        
        int cell = tid;
        Kokkos::Random_XorShift64_Pool<ExecutionSpace>::generator_type rand_gen = _r_p.get_state();    
        
        compute_test(rand_gen, garbage, size, div);
        _r_p.free_state(rand_gen);
    };    
    
    //TODO: prettify output
    //std::string headers[3] = {"Number of Cells", "Hier Time", "Par Time"};
    //std::cout << std::setw(headers[0].size()) << headers[0];
    
    {std::string title = "KOKKOS Hierarchical Tests";
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
    
    {std::string title = "KOKKOS Parallel For Tests";
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

    #ifdef KOKKOS_ENABLE_OPENMP
    #ifndef KOKKOS_ENABLE_CUDA
    {std::string title = "OMP Parallel For Tests";
    std::cout << title << std::endl;;
    using NUM = size_t;
    for(int i = 0; i <= 4; i++) {
        
        int threads = pow(2, i);    
        omp_set_num_threads(threads);
        
        NUM size = 1000000;
        NUM vals[size];


        auto start = std::chrono::high_resolution_clock::now();
        //set the values 
        #pragma omp parallel for

        for(NUM k = 0; k < threads; k++) {
            Kokkos::Random_XorShift64_Pool<ExecutionSpace>::generator_type rand_gen = _r_p.get_state(); 
            compute_test(rand_gen, garbage, size, div);
            _r_p.free_state(rand_gen);
        }
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration<double>(stop - start);
        std::cout << threads << " ran in " << duration.count() << " seconds\n";
    }std::cout << "\n\n";}
    #endif
    #endif
    return 0;
}
