#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <iostream>
#include <string>
#include <chrono>

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
    //seems to be where it maxes on GPU? (MOBILE GTX1060)
    int num_cells = 320; 
    double DELTA = 0.1;
    int NUM_INTERNAL_STEPS = 1000000;
    double DELTA_DELTA_T = DELTA / NUM_INTERNAL_STEPS;

    TestPolicy test_policy(num_cells, Kokkos::AUTO);

    const auto& _r_p = rand_pool;

    //could be replaced by scratch pad memory I think
    //Motivation: shared data between threads in a team
    //really, only the random sample needs to be single, but
    //it's easier on the brain if redundant work is not repeated
    Kokkos::View<double*> exp_sample("Exp Samples", num_cells);
    Kokkos::View<double*> tau("Tau", num_cells); 
    Kokkos::View<double*> delta_t("Delta_t", num_cells);

    auto test_hier = KOKKOS_LAMBDA(TestPolicy::member_type team_member) {
        
        int cell = team_member.league_rank();
        int t_r = team_member.team_rank();

        //Each team runs it's own shared simulation, so we need to have
        //the starting time set by only one thread and block until we are done
        Kokkos::single(Kokkos::PerTeam(team_member), [=] () {
            delta_t(cell) = 0.0;    
        });
        team_member.team_barrier();
        
        //All threads need to run this loop to meet the TeamThreadRange kernel
        while(delta_t(cell) < DELTA) {
            //Each team needs only one exponential sample, so we use
            //a single threads to set it and block all others from continuing
            //unitl we are done
            Kokkos::single(Kokkos::PerTeam(team_member), [=] () {
                double uniform_sample = uniform(_r_p, 0.0, 1.0);
                exp_sample(cell) = -log(1-uniform_sample);
                tau(cell) = 0.0;
                //printf("Exp Sample in Cell %d on thread %d is %f\n", cell, t_r, exp_sample(cell));
            });
            team_member.team_barrier();
            int counter = 0; 
            //All threads need to run this loop to meet the TeamThreadRange kernel
            while(delta_t(cell) < DELTA && tau(cell) < exp_sample(cell)) {
                
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
                Kokkos::single(Kokkos::PerTeam(team_member), [=] () {
                    tau(cell) += uniform(_r_p, 0.0, 0.1); //temporary placeholder of step (3)
                    //Advance inner time
                    delta_t(cell) += DELTA_DELTA_T;
                    //printf("Cell %d Step %d\n", cell, counter);
                });
                team_member.team_barrier();
                counter++;
                Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, 0, 2), [&](const int j) {
                    //printf("In Cell %d Inside loop %d on count %d\n", cell, i, j);
                });
                team_member.team_barrier();
            }
            
            //If we determined an event has occured, we need to block all other threads
            //and do some work(it's possible this may not be a fully single thread process)
            Kokkos::single(Kokkos::PerTeam(team_member), [=] () {
                if (tau(cell) > exp_sample(cell)) {
                    //printf("An event has occured in cell %d\n", cell);
                }
            });
            team_member.team_barrier();    
        }
        if(t_r == 0)
            printf("Cell %d done!\n", cell);
    };
    
    Kokkos::Timer timer;
    Kokkos::parallel_for("Test", test_policy, test_hier);
    Kokkos::fence();
    auto time = timer.seconds();
    std::cout << "Time: " << time << "\n";
    
 
    return 0;
}
