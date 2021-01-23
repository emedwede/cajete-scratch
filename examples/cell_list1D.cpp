#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>
#include <iostream>

template<class SliceType>
void print(SliceType& slice) {
    std::cout << "[";
    for(auto i = 0; i < slice.size(); i++) {
        std::cout << " " << slice(i) << " ";
    } std::cout << "]\n";
}

KOKKOS_INLINE_FUNCTION
int locatePoint(int x) {
    if(x < 3)
        return 0;
    else if(x >= 3 && x < 6) 
        return 1;
    else 
        return 2;
}

int main(int argc, char *argv[]) {

    //Initialize the kokkos runtime
    Kokkos::ScopeGuard scope_guard(argc, argv);
    
    printf("On Kokkos execution space %s\n", 
         typeid(Kokkos::DefaultExecutionSpace).name());
    
    using MemorySpace = Kokkos::DefaultExecutionSpace::memory_space;
    using ExecutionSpace = Kokkos::DefaultExecutionSpace;
    using DeviceType = Kokkos::Device<ExecutionSpace, MemorySpace>;
   
    using NodeTypes = Cabana::MemberTypes<int>;
        
    using particle_t_d = Cabana::AoSoA<NodeTypes, DeviceType>;
    using particle_t_h = particle_t_d::host_mirror_type;
    
    //number of particles
    int n_p = 12;

    //create particle aosoas
    particle_t_d particles_d("Particles on Device", n_p);
    particle_t_h particles_h("Particles on Host", n_p);

    //slice the particle aosoas
    auto positions_d = Cabana::slice<0>(particles_d);
    auto positions_h = Cabana::slice<0>(particles_h);

    //Preshuffled by me to avoid extra code
    int shuffled_points[n_p] = {8, 10,  4,  5, 11,  6,  2,  9,  7,  3,  1,  0};

    //set particles on host
    for(auto i = 0; i < particles_h.size(); i++) {
        positions_h(i) = shuffled_points[i];
    }
    //copy host particles to device
    Cabana::deep_copy(particles_d, particles_h);
    
    print(positions_h);

    //Create the binning data
    std::size_t ncell = 3;

    using view1D_d_t = Kokkos::View<int*, MemorySpace>; 
    using view1D_h_t = view1D_d_t::HostMirror;

    view1D_d_t counts(Kokkos::view_alloc(Kokkos::WithoutInitializing, "counts"),ncell);
    view1D_d_t offsets(Kokkos::view_alloc(Kokkos::WithoutInitializing, "offsets"),ncell);
    view1D_d_t permute(Kokkos::view_alloc(Kokkos::WithoutInitializing, "permute"),n_p);
    
    view1D_h_t counts_h = Kokkos::create_mirror_view(counts);
    view1D_h_t offsets_h = Kokkos::create_mirror_view(offsets);
    view1D_h_t permute_h = Kokkos::create_mirror_view(permute);

    Kokkos::RangePolicy<ExecutionSpace> particle_range_policy(0, n_p);

    auto counts_sv = Kokkos::Experimental::create_scatter_view( counts );
    
    //cell count function
    auto cell_count = KOKKOS_LAMBDA(const std::size_t p) {
        int cell_id = locatePoint(positions_d(p));
        auto counts_data = counts_sv.access();
        counts_data(cell_id) += 1;
    };
    
    Kokkos::parallel_for("Build cell list cell count", particle_range_policy, cell_count);
    Kokkos::fence();
    Kokkos::Experimental::contribute(counts, counts_sv);

    //compute offsets
    Kokkos::RangePolicy<ExecutionSpace> cell_range_policy(0, ncell);
    auto offset_scan = KOKKOS_LAMBDA(const size_t c, 
                                     int& update, 
                                     const bool final_pass) 
    {
        if(final_pass)
            offsets(c) = update;
        update += counts(c);
    };

    Kokkos::parallel_scan("Build cell list offset scan", cell_range_policy, offset_scan);
    Kokkos::fence();

    //Reset counts
    Kokkos::deep_copy(counts, 0);

    //create the permute vector, i.e. our indirection cell list
    auto create_permute = KOKKOS_LAMBDA(const size_t p) 
    {
        int cell_id = locatePoint(positions_d(p));
        int c = Kokkos::atomic_fetch_add( &counts(cell_id), 1 );
        permute(offsets(cell_id)+c) = p;
    };

    Kokkos::parallel_for("Build permute list", particle_range_policy, create_permute);

    Kokkos::deep_copy(permute_h, permute);
    Kokkos::deep_copy(counts_h, counts);
    Kokkos::deep_copy(offsets_h, offsets);

    std::cout << "\nCounts: [";
    for(auto i = 0; i < counts_h.size(); i++) {
        std::cout << " " << counts_h(i) << " ";
    } std::cout << "]\n";
    
    std::cout << "\nOffsets: [";
    for(auto i = 0; i < offsets_h.size(); i++) {
        std::cout << " " << offsets_h(i) << " ";
    } std::cout << "]\n";

    std::cout << "\nPermute Vector: [";
    for(auto i = 0; i < permute_h.size(); i++) {
        std::cout << " " << permute_h(i) << " ";
    } std::cout << "]\n";
    return 0;
}
