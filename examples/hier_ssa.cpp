#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <iostream>
#include <string>
#include <chrono>
#include <sstream>

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
    std::ostringstream msg;
    printf("On Kokkos execution space %s\n", typeid(Kokkos::DefaultExecutionSpace).name());
    msg << "{" << std::endl;
    if (Kokkos::hwloc::available()) {
        msg << "hwloc( NUMA[" << Kokkos::hwloc::get_available_numa_count()
            << "] x CORE[" << Kokkos::hwloc::get_available_cores_per_numa()
            << "] x HT[" << Kokkos::hwloc::get_available_threads_per_core() << "] )"
            << std::endl;
    }

    #if defined(KOKKOS_ENABLE_CUDA)
        Kokkos::Cuda::print_configuration(msg);
    #endif
    msg << "}" << std::endl;
    std::cout << msg.str();
    
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
    auto test_hier = KOKKOS_LAMBDA(TestPolicy::member_type team_member) {
        auto rand_gen = _r_p.get_state(); 
        int cell = team_member.league_rank();
        int t_r = team_member.team_rank();
        //shared_double test(team_member.team_scratch(0), team_member.team_size()); 
        //Kokkos::View<double, Kokkos::MemoryUnmanaged> test_data(team_member.team_shmem());

        //Each team runs it's own shared simulation, so we could  have
        //the starting time set by only one thread and block until we are done
        //But, it doesn't really matter, so we'll give each local thread a copy
        double delta_t, exp_sample, tau, cell_propensity; int events;
        delta_t = 0.0; events = 0; 

        //All threads need to run this loop to meet the TeamThreadRange kernel
        while(delta_t < DELTA) {
            //reset tau
            tau = 0.0;

            //Each team needs only one exponential sample, so we use single to broadcast
            //in practice single is actually implemented as something like:
            //if(team.team_rank()==0) { lambda(value); } team.team_broadcast(value,0);
            Kokkos::single(Kokkos::PerTeam(team_member), [&] (double &sample) {
                double uniform_sample = rand_gen.drand(0.0, 1.0);
                sample = -log(1-uniform_sample);
            }, exp_sample);
            team_member.team_barrier();

            double counter = 0.0; //currently used to force the compiler not to optimize 
            
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
                
                cell_propensity = 0.0;
                Kokkos::parallel_reduce(
                        Kokkos::TeamThreadRange(team_member, 64),
                    [&] (const int pid, double& local_propensity) {
                    local_propensity += rand_gen.drand(0.1, 0.2);
                }, cell_propensity);
                team_member.team_barrier();
                
                //Combination of (3) and (4)
                tau += cell_propensity*DELTA_DELTA_T;
                delta_t += DELTA_DELTA_T;
                
                for(auto i = 0; i < 1; i++) {
                    double samp = rand_gen.drand(0.0, 1.0);
                    counter = i*delta_t*samp;
                }
                
                counter += 1.0; 
            }
            team_member.team_barrier();

            //If we determined an event has occured, we need to block all other threads
            //and do some work(it's possible this may not be a fully single thread process)
            Kokkos::single(Kokkos::PerTeam(team_member), [&] (int &value) {
                if (tau > exp_sample) {
                    //view are default alloc to zero
                    value++;
                }
            }, events);
            team_member.team_barrier();     
        }
        //if(t_r == 0) {
            //printf("%d events in cell %d\n", events, cell);
        //}
        rand_pool.free_state(rand_gen); 
    };
    
    int num_runs = 15;
    double run_times[num_runs+1];
    double num_cells[num_runs+1];
    for(int j = 0; j <= num_runs; j++) {
        int cells = pow(2, j);
        num_cells[j] = cells;
        TestPolicy test_policy(cells, Kokkos::AUTO);
        
        Kokkos::Timer timer;
        Kokkos::parallel_for("Test", test_policy, test_hier);
        Kokkos::fence();
        double time = timer.seconds();
        run_times[j] = time;
        printf("%d cells took %f seconds\n", cells, time);
    } 

    std::cout << "run_times = np.asarray([";
    for(int j = 0; j < num_runs+1; j++) {
        if(j < num_runs) {
            std::cout << run_times[j] << ", ";
        } else {
            std::cout << run_times[j] << "])\n";
        }
    }
    std::cout << "num_cells = np.asarray([";
    for(int j = 0; j < num_runs+1; j++) {
        if(j < num_runs) {
            std::cout << num_cells[j] << ", ";
        } else {
            std::cout << num_cells[j] << "])\n";
        }
    }

    return 0;
}
