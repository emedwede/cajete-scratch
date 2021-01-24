#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>
#include <iostream>
#include <string>

template<class SliceType>
void print(SliceType& slice, std::string msg="") {
    std::cout << msg << ": [";
    for(auto i = 0; i < slice.size(); i++) {
        std::cout << " " << slice(i) << " ";
    } std::cout << "]\n";
}


KOKKOS_INLINE_FUNCTION
int locatePoint(float x) {
    if(x < 3)
        return 0;
    else if(x >= 3 && x < 6) 
        return 1;
    else 
        return 2;
}

struct grid1D {
    float _min_x;
    float _max_x;
    float _dx;
    float _nx_g;
    float _cs;  //reaction radius cell size
    float _nx_l; //number of local xcells

    grid1D(float max_x, float min_x, float dx, float rr) 
        : _min_x(min_x), _max_x(max_x), _dx(dx), _cs(rr) 
    {
        _nx_g = cellsBetween(_max_x, _min_x, (1.0 / _dx) );
        //adjusted dx to accomodate an integer number of cells
        _dx   = ( _max_x - _min_x ) / _nx_g;  
        //simple in this case because we have uniform partition
        _nx_l = cellsBetween(_dx, 0.0, (1.0 / _cs ) );
        //Adjusted cell size, should always be _cs >= rr
        _cs   = ( _dx / _nx_l ); 
    }

    KOKKOS_INLINE_FUNCTION
    int cellsBetween(const float max, const float min, const float rdelta) const {
        return floor( (max - min) * rdelta);
    }

    KOKKOS_INLINE_FUNCTION
    int locatePointGlobal(float xp) const {
        int ic_g = cellsBetween(xp, _min_x, ( 1.0 / _dx ));
        return ( ic_g == _nx_g ) ? ic_g - 1 : ic_g;
    }

    KOKKOS_INLINE_FUNCTION
    int locatePointLocal(float xp) const {
        int ic_g = locatePointGlobal(xp);
        int ic_l = cellsBetween(xp, ic_g*_dx, _cs);
        return ic_l;
    }

};

int main(int argc, char *argv[]) {

    //Initialize the kokkos runtime
    Kokkos::ScopeGuard scope_guard(argc, argv);
    
    printf("On Kokkos execution space %s\n", 
         typeid(Kokkos::DefaultExecutionSpace).name());
    
    using MemorySpace = Kokkos::DefaultExecutionSpace::memory_space;
    using ExecutionSpace = Kokkos::DefaultExecutionSpace;
    using DeviceType = Kokkos::Device<ExecutionSpace, MemorySpace>;
   
    using NodeTypes = Cabana::MemberTypes<float>;
        
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
    float shuffled_points[n_p] = {0.5, 8.75, 1.5, 8.25, 2.5, 7.75, 3.5, 7.25, 4.5, 6.75, 5.5, 6.25};

    //set particles on host
    for(auto i = 0; i < particles_h.size(); i++) {
        positions_h(i) = shuffled_points[i];
    }
    //copy host particles to device
    Cabana::deep_copy(particles_d, particles_h);
    
    print(positions_h);

    grid1D grid(9.0, 0.0, 3.0, 1.0);
    
    //Create the binning data
    int ncell = grid._nx_g;

    using view1D_d_t = Kokkos::View<int*, MemorySpace>; 
    using view1D_h_t = view1D_d_t::HostMirror;

    view1D_d_t counts(Kokkos::view_alloc(Kokkos::WithoutInitializing, "counts"),ncell);
    view1D_d_t offsets(Kokkos::view_alloc(Kokkos::WithoutInitializing, "offsets"),ncell);
    view1D_d_t permute(Kokkos::view_alloc(Kokkos::WithoutInitializing, "permute"),n_p);
    
    view1D_h_t counts_h = Kokkos::create_mirror_view(counts);
    view1D_h_t offsets_h = Kokkos::create_mirror_view(offsets);
    view1D_h_t permute_h = Kokkos::create_mirror_view(permute);

    Kokkos::RangePolicy<ExecutionSpace> particle_range_policy(0, n_p);

    /*auto counts_sv = Kokkos::Experimental::create_scatter_view( counts );
    
    //cell count function
    auto cell_count = KOKKOS_LAMBDA(const std::size_t p) {
        int cell_id = locatePoint(positions_d(p));
        auto counts_data = counts_sv.access();
        counts_data(cell_id) += 1;
    };
    
    Kokkos::parallel_for("Build cell list cell count", particle_range_policy, cell_count);
    Kokkos::fence();
    Kokkos::Experimental::contribute(counts, counts_sv);
    */

    //A less experimental and more straightforward way
    Kokkos::deep_copy(counts, 0);
    auto cell_count_atomic = KOKKOS_LAMBDA(const std::size_t p) {
        int cell_id = grid.locatePointGlobal(positions_d(p));
        Kokkos::atomic_increment(&counts(cell_id));
    };
    Kokkos::parallel_for("Build cell list cell count", particle_range_policy, cell_count_atomic);
    Kokkos::fence();
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
        int cell_id = grid.locatePointGlobal(positions_d(p));
        int c = Kokkos::atomic_fetch_add( &counts(cell_id), 1 );
        permute(offsets(cell_id)+c) = p;
    };

    Kokkos::parallel_for("Build permute list", particle_range_policy, create_permute);

    Kokkos::deep_copy(permute_h, permute);
    Kokkos::deep_copy(counts_h, counts);
    Kokkos::deep_copy(offsets_h, offsets);

    print(counts_h, "\nCounts");
    print(offsets_h, "\nOffsets");
    print(permute_h, "\nPermute");

    return 0;
}
