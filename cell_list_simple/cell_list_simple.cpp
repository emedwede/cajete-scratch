#include<iostream>

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>

#include "grid.hpp"

//Linked Cell List

namespace Cajete {

template<class DeviceType>
class ParticleArray {
    public:
        enum ParticleFields {position=0, id};
        using device = DeviceType;
        using memory_space = typename device::memory_space;
        using execution_space = typename device::execution_space;
        
        using NodeTypes = Cabana::MemberTypes<double[3], size_t>;
        
        using p_array_t_d = Cabana::AoSoA<NodeTypes, device>;
        using positions_t_d = typename p_array_t_d::template member_slice_type<position>;
        using ids_t_d = typename p_array_t_d::template member_slice_type<id>;

        using p_array_t_h = typename p_array_t_d::host_mirror_type;
        using positions_t_h = typename p_array_t_h::template member_slice_type<position>;
        using ids_t_h = typename p_array_t_h::template member_slice_type<id>;

        p_array_t_d particles_d;
        positions_t_d positions_d;
        ids_t_d ids_d;

        p_array_t_h particles_h;
        positions_t_h positions_h;
        ids_t_h ids_h;
        
        struct TestFillTag{};
        using TestFillTagPolicy = Kokkos::RangePolicy<execution_space, TestFillTag>;

        KOKKOS_INLINE_FUNCTION
        void operator()(const TestFillTag&, const int i) const {
            for(int j = 0; j < 3; j++) positions_d(i, j) = i;
            ids_d(i) = i;
        }

        void test_fill() {
            reslice();
            TestFillTagPolicy test_fill_policy(0, particles_d.size());
            Kokkos::parallel_for("Test Fill Particles Array", test_fill_policy, *this);
            Kokkos::fence();
        }
        
        ParticleArray(size_t n_p) : particles_d("DeviceParticles", n_p)
                                  , positions_d(Cabana::slice<position>(particles_d))
                                  , ids_d(Cabana::slice<id>(particles_d))
                                  , particles_h("HostParticles", n_p)
                                  , positions_h(Cabana::slice<position>(particles_h))
                                  , ids_h(Cabana::slice<id>(particles_h))
        {
            test_fill();
        }

        size_t size() {return particles_d.size();}

        void reslice() {
            positions_d = Cabana::slice<position>(particles_d);
            ids_d = Cabana::slice<id>(particles_d);
            positions_h = Cabana::slice<position>(particles_h);
            ids_h = Cabana::slice<id>(particles_h);
        }

        void copy_host_to_device() {
            Cabana::deep_copy(particles_d, particles_h);
        }

        void copy_device_to_host() {
            Cabana::deep_copy(particles_h, particles_d);
        }

        void print() {
            copy_device_to_host();
            reslice();
            std::cout << "Positions and ids: \n";
            for(auto i = 0; i < particles_h.size(); i++) {
                std::cout << "{ ";
                for(int j = 0; j < 3; j++) {
                    std::cout << positions_h(i, j) << " ";
                } std::cout << "}, " << ids_h(i) << std::endl;
            }
        }

};

template<class DeviceType>
class CellComplex {
    public:
        
        using device = DeviceType;
        using memory_space = typename device::memory_space;
        using execution_space = typename device::execution_space;
        
        using local_grid_d_t = Kokkos::View<Cajete::CartesianGrid2D<double>*, device>;
        using local_grid_h_t = typename local_grid_d_t::HostMirror;
        local_grid_d_t grid_view_d;
        local_grid_h_t grid_view_h;
        Cajete::BrickGrid2D<double> _grid;
        
        struct BuildLocalGridTag{};
        using BuildLocalGridTagPolicy = Kokkos::RangePolicy<execution_space, BuildLocalGridTag>;

        KOKKOS_INLINE_FUNCTION
        void operator()(const BuildLocalGridTag&, const int i) const {
            grid_view_d(i).init(0.0, 0.0, 3.0, 3.0, 1.0, 1.0);
            int ncell = grid_view_d(i).totalNumCells();
            printf("GPU : %d : %d\n", i, ncell);

            int ic = 0, jc = 0;
            grid_view_d(i).locatePoint(3.0, 3.0, ic, jc);
            int cardinal = grid_view_d(i).cardinalCellIndex(ic, jc);
            printf("Point in cell %d\n", cardinal);
        }

        void build_local_grid() {
            BuildLocalGridTagPolicy build_local_grid_policy(0, grid_view_d.size());
            Kokkos::parallel_for("Test Build Local Grid", build_local_grid_policy, *this);
            Kokkos::fence();
        }

        CellComplex() {}

        CellComplex(double grid_min[2], double grid_max[2], double grid_delta[2]) 
            : _grid(grid_min[0], grid_min[1],
                    grid_max[0], grid_max[1],
                    grid_delta[0], grid_delta[1]) 
        {
            grid_view_d = local_grid_d_t("local", _grid.totalNumCells());
            grid_view_h = Kokkos::create_mirror_view(grid_view_d);

            build_local_grid();
            Kokkos::deep_copy(grid_view_h, grid_view_d);
            for(int i = 0; i < 5; i++) {
                //int ncell = grid_view_h(i)._nx*grid_view_h(i)._ny;
                int ncell = grid_view_h(i).totalNumCells();
                printf("CPU : %d : %d\n", i, ncell);
            } 
            std::cout << "Constructing cell complex\n";
        }

};

template<class DeviceType>
class LLCellList {
    public:
        using device_type = DeviceType;
        using memory_space = typename device_type::memory_space;
        using execution_space = typename device_type::execution_space;
        using size_type = typename memory_space::size_type;
        using OffsetView = Kokkos::View<size_type*, device_type>;

        LLCellList() {} //Default
        
        template<class SliceType>
        LLCellList(
            SliceType positions, const typename SliceType::value_type grid_delta[2],
            const typename SliceType::value_type grid_min[2],
            const typename SliceType::value_type grid_max[2],
            typename std::enable_if<( Cabana::is_slice<SliceType>::value ), int>::type* = 0)
            : _grid(grid_min[0], grid_min[1], 
                    grid_max[0], grid_max[1], 
                    grid_delta[0], grid_delta[1])
        {
            build(positions, 0, positions.size());
        }

        template<class SliceType>
        void build(SliceType positions, const std::size_t begin, 
                const std::size_t end) 
        {
            std::cout << "\n----------\nBuilding\n----------\n";
        }

    private:
        Cajete::BrickGrid2D<double> _grid;
};

}

int main(int argc, char *argv[]) {
    
    Kokkos::ScopeGuard scope_guard(argc, argv);

    using MemorySpace = Kokkos::DefaultExecutionSpace::memory_space;
    using ExecutionSpace = Kokkos::DefaultExecutionSpace;
    using DeviceType = Kokkos::Device<ExecutionSpace, MemorySpace>;
    
    Cajete::ParticleArray<DeviceType> p_array(10);
    std::cout << "Size: " << p_array.size() << std::endl;
    p_array.print();

    double grid_delta[2] = {1.0, 1.0};
    double grid_min[2] = {0.0, 0.0};
    double grid_max[2] = {2.0, 2.0};

    Cajete::LLCellList<DeviceType> cell_list(p_array.positions_d, grid_delta, grid_min, grid_max);

    Cajete::CellComplex<DeviceType> cell_complex(grid_min, grid_max, grid_delta);
}
