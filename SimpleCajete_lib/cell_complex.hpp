#ifndef __CAJETE_CELL_COMPLEX_HPP
#define __CAJETE_CELL_COMPLEX_HPP

#include<iostream>

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>

#include "grid.hpp"

namespace Cajete {

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
        Cajete::BrickGrid2D<double> _global_grid;
        
        struct BuildLocalGridTag{};
        using BuildLocalGridTagPolicy = Kokkos::RangePolicy<execution_space, BuildLocalGridTag>;

        KOKKOS_INLINE_FUNCTION
        void operator()(const BuildLocalGridTag&, const int i) const {
            double x_min, y_min, x_max, y_max;
            _global_grid.minMaxCellCorners(2, 1, x_min, y_min, x_max, y_max);
            printf("%f %f %f %f\n", x_min, y_min, x_max, y_max);
            grid_view_d(i).init(0.0, 0.0, 2.0, 2.0, 1.0, 1.0);
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
            : _global_grid(grid_min[0], grid_min[1],
                    grid_max[0], grid_max[1],
                    grid_delta[0], grid_delta[1]) 
        {
            grid_view_d = local_grid_d_t("local", _global_grid.totalNumCells());
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

}
#endif
