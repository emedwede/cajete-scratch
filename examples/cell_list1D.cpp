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
        int ic_l = cellsBetween(xp, ic_g*_dx, 1.0 / _cs);
        return (ic_l == _nx_l) ? ic_l - 1 : ic_l;
    }

};

template<class DeviceType>
struct p_array1D {

    using NodeTypes = Cabana::MemberTypes<float>;
    
    using particles_t_d = Cabana::AoSoA<NodeTypes, DeviceType>;
    using positions_t_d = typename particles_t_d::template member_slice_type<0>;

    using particles_t_h = typename particles_t_d::host_mirror_type;
    using positions_t_h = typename particles_t_h::template member_slice_type<0>;

    particles_t_d particles_d;
    particles_t_h particles_h;

    positions_t_d positions_d;
    positions_t_h positions_h;

    p_array1D(int n_p) 
        : particles_d("Particles on Device", n_p)
        , particles_h("Particles on Host", n_p)
        , positions_d(Cabana::slice<0>(particles_d))
        , positions_h(Cabana::slice<0>(particles_h))
    {
        
    }

    void reslice() {
        positions_d = Cabana::slice<0>(particles_d);
        positions_h = Cabana::slice<0>(particles_h);
    }

    void copy_host_to_device() {
        Cabana::deep_copy(positions_d, positions_h);
        reslice();
    }

    void copy_device_to_host() {
        Cabana::deep_copy(positions_h, positions_d);
        reslice();
    }

};

//Preset the data for our example
template<class DeviceType>
void preset(p_array1D<DeviceType>& p_array) {
    //Preshuffled by me to avoid extra code
    float shuffled_points[12] = {0.5, 8.75, 1.5, 8.25, 2.5, 7.75, 3.5, 7.25, 4.5, 6.75, 5.5, 6.25};    
    //set particles on host
    for(auto i = 0; i < 12; i++) {
        p_array.positions_h(i) = shuffled_points[i];
    }
    p_array.copy_host_to_device();
}

template <class DeviceType>
struct CellList {
    using device = DeviceType;
    using memory_space = typename device::memory_space;
    using execution_space = typename device::execution_space;

    using view1D_d_t = Kokkos::View<int*, memory_space>; 
    using view1D_h_t = typename view1D_d_t::HostMirror;
    
    grid1D grid;

    template<class SliceType> 
    CellList (SliceType positions, 
              float min, float max, 
              float dx, float rr) 
              : grid(max, min, dx, rr)
    { build(positions); }

    template<class SliceType>
    void build(SliceType positions) {
        //Get a copy of grid for cuda
        auto const& _grid = grid;
        int ncell = _grid._nx_g;
        int n_p = positions.size(); 


        view1D_d_t counts_d(Kokkos::view_alloc(Kokkos::WithoutInitializing, "counts"),ncell);
        view1D_d_t offsets_d(Kokkos::view_alloc(Kokkos::WithoutInitializing, "offsets"),ncell);
        view1D_d_t permute_d(Kokkos::view_alloc(Kokkos::WithoutInitializing, "permute"),n_p);
        
        view1D_h_t counts_h = Kokkos::create_mirror_view(counts_d);
        view1D_h_t offsets_h = Kokkos::create_mirror_view(offsets_d);
        view1D_h_t permute_h = Kokkos::create_mirror_view(permute_d);
        
        Kokkos::RangePolicy<execution_space> particle_range_policy(0, n_p);
        
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
        Kokkos::deep_copy(counts_d, 0);
        auto cell_count_atomic = KOKKOS_LAMBDA(const std::size_t p) {
            int cell_id = _grid.locatePointGlobal(positions(p));
            Kokkos::atomic_increment(&counts_d(cell_id));
        };
        
        Kokkos::parallel_for("Build cell list cell count", particle_range_policy, cell_count_atomic);
        Kokkos::fence();
        
        //compute offsets
        Kokkos::RangePolicy<execution_space> cell_range_policy(0, ncell);
        auto offset_scan = KOKKOS_LAMBDA(const size_t c, 
                                         int& update, 
                                         const bool final_pass) 
        {
            if(final_pass)
                offsets_d(c) = update;
            update += counts_d(c);
        };

        Kokkos::parallel_scan("Build cell list offset scan", cell_range_policy, offset_scan);
        Kokkos::fence();

        //Reset counts
        Kokkos::deep_copy(counts_d, 0);

        //create the permute vector, i.e. our indirection cell list
        auto create_permute = KOKKOS_LAMBDA(const size_t p) 
        {
            int cell_id = _grid.locatePointGlobal(positions(p));
            int c = Kokkos::atomic_fetch_add( &counts_d(cell_id), 1 );
            permute_d(offsets_d(cell_id)+c) = p;
        };

        Kokkos::parallel_for("Build permute list", particle_range_policy, create_permute);

        Kokkos::deep_copy(permute_h, permute_d);
        Kokkos::deep_copy(counts_h, counts_d);
        Kokkos::deep_copy(offsets_h, offsets_d); 
        
        print(counts_h, "\nCounts");
        print(offsets_h, "\nOffsets");
        print(permute_h, "\nPermute");
    }
        
};


template <class DeviceType>
struct LLCellList {
    using device = DeviceType;
    using memory_space = typename device::memory_space;
    using execution_space = typename device::execution_space;

    using view1D_d_t = Kokkos::View<int*, memory_space>; 
    using view1D_h_t = typename view1D_d_t::HostMirror;
    
    grid1D grid;
    
    view1D_d_t _counts_d;
    view1D_h_t _counts_h;
    view1D_d_t _offsets_d;
    view1D_h_t _offsets_h;
    
    view1D_d_t _counts_l_d;
    view1D_h_t _counts_l_h;
    view1D_d_t _offsets_l_d;
    view1D_h_t _offsets_l_h;
    view1D_d_t _permute_l_d;
    view1D_h_t _permute_l_h;

    template<class SliceType> 
    LLCellList (SliceType positions, 
              float min, float max, 
              float dx, float rr) 
              : grid(max, min, dx, rr)
    { build(positions); }

    template<class SliceType>
    void build(SliceType positions) {
        //Get a copy of grid for cuda
        auto const& _grid = grid;
        int ncell = _grid._nx_g; 
        int ncell_l = _grid._nx_l; //Number of local cells
        int n_p = positions.size(); 


        view1D_d_t counts_d(Kokkos::view_alloc(Kokkos::WithoutInitializing, "counts"),ncell);
        view1D_d_t offsets_d(Kokkos::view_alloc(Kokkos::WithoutInitializing, "offsets"),ncell);
        
        view1D_h_t counts_h = Kokkos::create_mirror_view(counts_d);
        view1D_h_t offsets_h = Kokkos::create_mirror_view(offsets_d);
       
        view1D_d_t counts_l_d(Kokkos::view_alloc(Kokkos::WithoutInitializing, "counts"),ncell*ncell_l);
        view1D_d_t offsets_l_d(Kokkos::view_alloc(Kokkos::WithoutInitializing, "offsets"),ncell*ncell_l);
        view1D_d_t permute_l_d(Kokkos::view_alloc(Kokkos::WithoutInitializing, "permute"),n_p);
        
        view1D_h_t counts_l_h = Kokkos::create_mirror_view(counts_l_d);
        view1D_h_t offsets_l_h = Kokkos::create_mirror_view(offsets_l_d);
        view1D_h_t permute_l_h = Kokkos::create_mirror_view(permute_l_d);
 
        Kokkos::RangePolicy<execution_space> particle_range_policy(0, n_p);
        
        //A less experimental and more straightforward way
        Kokkos::deep_copy(counts_d, 0);
        auto cell_count_atomic = KOKKOS_LAMBDA(const std::size_t p) {
            int cell_id = _grid.locatePointGlobal(positions(p));
            Kokkos::atomic_increment(&counts_d(cell_id));
        };
        
        Kokkos::parallel_for("Build cell list cell count", particle_range_policy, cell_count_atomic);
        Kokkos::fence();
       
        Kokkos::deep_copy(counts_l_d, 0);
        auto local_cell_count_atomic = KOKKOS_LAMBDA(const std::size_t p) {
            int cell_id = _grid.locatePointGlobal(positions(p));
            int cell_id_local = _grid.locatePointLocal(positions(p));
            Kokkos::atomic_increment(&counts_l_d(cell_id*ncell+cell_id_local));
        };
        
        Kokkos::parallel_for("Build cell list local cell count", particle_range_policy, local_cell_count_atomic);
        Kokkos::fence();
        

        //compute offsets
        Kokkos::RangePolicy<execution_space> cell_range_policy(0, ncell);
        auto offset_scan = KOKKOS_LAMBDA(const size_t c, 
                                         int& update, 
                                         const bool final_pass) 
        {
            if(final_pass)
                offsets_d(c) = update;
            update += counts_d(c);
        };

        Kokkos::parallel_scan("Build cell list offset scan", cell_range_policy, offset_scan);
        Kokkos::fence();

        //compute offsets
        Kokkos::RangePolicy<execution_space> cell_range_policy_local(0, ncell*ncell_l);
        auto offset_scan_local = KOKKOS_LAMBDA(const size_t c, 
                                         int& update, 
                                         const bool final_pass) 
        {
            if(final_pass)
                offsets_l_d(c) = update;
            update += counts_l_d(c);
        };

        Kokkos::parallel_scan("Build cell list offset scan", cell_range_policy_local, offset_scan_local);
        Kokkos::fence();

        //Reset local counts
        Kokkos::deep_copy(counts_l_d, 0);

        //create the permute vector on the local scale, i.e. our indirection cell list
        auto create_permute_local = KOKKOS_LAMBDA(const size_t p) 
        {
            int cell_id = _grid.locatePointGlobal(positions(p));
            int local_cell_id = _grid.locatePointLocal(positions(p));
            int c = Kokkos::atomic_fetch_add( &counts_l_d(cell_id*ncell+local_cell_id), 1 );
            permute_l_d(offsets_l_d(cell_id*ncell+local_cell_id)+c) = p;
        };

        Kokkos::parallel_for("Build local permute list", particle_range_policy, create_permute_local);

        Kokkos::deep_copy(counts_h, counts_d);
        Kokkos::deep_copy(offsets_h, offsets_d); 
        Kokkos::deep_copy(counts_l_h, counts_l_d);
        Kokkos::deep_copy(offsets_l_h, offsets_l_d);
        Kokkos::deep_copy(permute_l_h, permute_l_d);

        _counts_d = counts_d;
        _counts_h = counts_h;
        _offsets_d = offsets_d;
        _offsets_h = offsets_h;
        _counts_l_d = counts_l_d;
        _counts_l_h = counts_l_h;
        _offsets_l_d = offsets_l_d;
        _offsets_l_h = offsets_l_h;
        _permute_l_d = permute_l_d;
        _permute_l_h = permute_l_h;
    }   

    void show() {
        print(_counts_h, "\nGlobal Cell Counts");
        print(_offsets_h, "\nGlobal Cell Offsets");
        print(_counts_l_h, "\nLocal Counts");
        print(_offsets_l_h, "\nLocal Offsets");
        print(_permute_l_h, "\nLocal Permute");
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
   
    int n_p = 12; //number of particles
    p_array1D<DeviceType> particles(n_p);
    preset(particles);
    print(particles.positions_h);


    auto my_slice = Cabana::slice<0>(particles.particles_d); 
    CellList<DeviceType> cell_list_global(my_slice, 0.0, 9.0, 3.0, 1.0);
    LLCellList<DeviceType> cell_list_local(my_slice, 0.0, 9.0, 3.0, 1.0);
    cell_list_local.show();
    
    //needed for correct cuda capture
    auto const& _cell_list_local = cell_list_local;

    using NeighborAccessPolicy = 
        Kokkos::TeamPolicy<ExecutionSpace, 
        Kokkos::IndexType<int>, 
        Kokkos::Schedule<Kokkos::Dynamic>>;

    int ncells_g = cell_list_local._counts_d.size();
    NeighborAccessPolicy access_policy(ncells_g, Kokkos::AUTO);//, 4);
    auto neighbor_access = KOKKOS_LAMBDA(NeighborAccessPolicy::member_type team) {
        int cell_g  = team.league_rank();
        int t_r = team.team_rank();
        if(t_r == 0) {
            int l_r = team.league_rank();
            int t_s = team.team_size();
            int c = _cell_list_local._counts_d(cell_g);
            printf("Global Cell %d has team size %d\n", l_r, t_s);
            printf("Golbal Cell %d has %d particles\n", l_r, c);
        }

        int ncell_l = _cell_list_local.grid._nx_l;
        int np_l = _cell_list_local._counts_d(cell_g);
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, 0, np_l), [&](const int bi) {
            int l_r = team.league_rank();
            printf("Inside Global Cell %d visiting Particle %d\n", l_r, bi);
        });
    };
    Kokkos::parallel_for("Access LLCellList Nbrs", access_policy, neighbor_access);

    return 0;
}
