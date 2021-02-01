#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <iostream>
#include <string>
#include <chrono>

using MemorySpace = Kokkos::DefaultExecutionSpace::memory_space;
using ExecutionSpace = Kokkos::DefaultExecutionSpace;
//using ExecutionSpace = Kokkos::Serial;//Kokkos::DefaultExecutionSpace;
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
    //320 cells seems to be where it maxes on GPU? (MOBILE GTX1060) 
    double DELTA = 0.1;
    int NUM_INTERNAL_STEPS = 5000;
    double DELTA_DELTA_T = DELTA / NUM_INTERNAL_STEPS;

    
    const auto& _r_p = rand_pool;

    //could be replaced by scratch pad memory I think
    //Motivation: shared data between threads in a team
    //really, only the random sample needs to be single, but
    //it's easier on the brain if redundant work is not repeated
    int num_cells = 200;
    Kokkos::View<double*> _exp_sample("Exp Samples", num_cells);
    Kokkos::View<double*> _tau("Tau", num_cells); 
    Kokkos::View<double*> _delta_t("Delta_t", num_cells);
    Kokkos::View<double*> _events("Events", num_cells); 
    typedef ExecutionSpace::scratch_memory_space ScratchSpace;
    typedef Kokkos::View<double, ScratchSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> shared_double;
    auto test_hier = KOKKOS_LAMBDA(TestPolicy::member_type team_member) {
        auto rand_gen = _r_p.get_state(); 
        int cell = team_member.league_rank();
        int t_r = team_member.team_rank();
        //shared_double test(team_member.team_scratch(0)); 
       
        //Each team runs it's own shared simulation, so we could  have
        //the starting time set by only one thread and block until we are done
        //But, it doesn't really matter, so we'll give each local thread a copy
        double delta_t = 0.0; 
        
        //equivalent to Kokkos::single
        if(t_r == 0) {
            //make sure no events are tracked yet
            _events(cell) = 0;
        }
        team_member.team_barrier();

        //All threads need to run this loop to meet the TeamThreadRange kernel
        while(delta_t < DELTA) {
            //Each team needs only one exponential sample, so we use
            //a single threads to set it and block all others from continuing
            //until we are done
            Kokkos::single(Kokkos::PerTeam(team_member), [&] () {
                double uniform_sample = rand_gen.drand(0.0, 1.0);
                //uniform_sample = 0.5;
                _exp_sample(cell) = -log(1-uniform_sample);
                _tau(cell) = 0.0;
            });
            team_member.team_barrier();
            double counter = 0.0; //currently used to force the compiler not to optimize 
            
            //set the local tau and cache the exp sample
            double tau = _tau(cell);
            double exp_sample = _exp_sample(cell);
            
            //All threads need to run this loop to meet the TeamThreadRange kernel
            while(delta_t < DELTA && tau < exp_sample) {  
                //STEP (1) : solve system of particle odes
                //BARRIER: allow for updated particles to be used in (2) 

                //STEP (2) : summ all particle-particle interaction probabilities 
                //           into a total propensity, potentially with a redue?
                //BARRIER: to allow for propensity to be summed

                //STEP (3) : use forward euler to solve the basic tau ODE
                //           should be done by a single thread
                //tau += total_propensity*DELTA_DELTA_T
                
                //STEP (4) : advance the loop timer, really, only one thread needs 
                //           to do this
                //BARRIER

                //Combination of (3) and (4)
                Kokkos::single(Kokkos::PerTeam(team_member), [&] () {
                    double tau_samp = rand_gen.drand(0.001, 0.01);
                    _tau(cell) += tau_samp;
                    //printf("Tau %f\n", tau_samp);
                    //_tau(cell) += 0.001;
                });
                team_member.team_barrier();
                tau = _tau(cell);
                delta_t += DELTA_DELTA_T;
                
                for(auto i = 0; i < 2500; i++) {
                    double samp = rand_gen.drand(0.0, 1.0);
                    counter = i*delta_t*samp;
                }
                
                counter += 1.0; 
            }
            team_member.team_barrier();

            //If we determined an event has occured, we need to block all other threads
            //and do some work(it's possible this may not be a fully single thread process)
            Kokkos::single(Kokkos::PerTeam(team_member), [&] () {
                if (_tau(cell) > _exp_sample(cell)) {
                    //view are default alloc to zero
                    _events(cell)++;
                }
            });
            team_member.team_barrier();     
        }
        if(t_r == 0) {
            int n_e = _events(cell);
            //printf("%d events in cell %d\n", n_e, cell);
        }
        rand_pool.free_state(rand_gen); 
    };
    for(int i = 0; i <= 0; i++) {
    for(int j = 0; j <= 4; j++) {
        int num_cells = pow(2, j);
        int num_threads = pow(2, i);
        TestPolicy test_policy(num_cells, Kokkos::AUTO);//num_threads);//Kokkos::AUTO);
        int team_size = test_policy.team_size();
        Kokkos::Timer timer;
        Kokkos::parallel_for("Test", test_policy, test_hier);
        Kokkos::fence();
        double time = timer.seconds();
        printf("%d cells %d threads took %f seconds\n", num_cells, team_size, time);
    }} 
 
    return 0;
}
