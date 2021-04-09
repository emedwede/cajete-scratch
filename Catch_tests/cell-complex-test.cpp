#include "catch.hpp"
#include "grid.hpp"
#include "particle_array.hpp"
#include "cell_complex.hpp"
#include "ll_cell_list.hpp"

TEST_CASE("Grid Set Builder", "[cell_complex_test]") 
{
    using MemorySpace = Kokkos::DefaultExecutionSpace::memory_space;
    using ExecutionSpace = Kokkos::DefaultExecutionSpace;
    using DeviceType = Kokkos::Device<ExecutionSpace, MemorySpace>;
    
    double grid_delta[2] = {1.0, 1.0};
    double grid_min[2] = {0.0, 0.0};
    double grid_max[2] = {2.0, 2.0};

    Cajete::StrataGridBuilder<DeviceType> grid_strata(grid_min, grid_max, grid_delta);

    REQUIRE(grid_strata._global_grid.totalNumCells() == 5);
    
    REQUIRE(grid_strata.grid_view_h(0).totalNumCells() == 4);

}

TEST_CASE("Cell Complex Init", "[cell_complex_test]") 
{
    using MemorySpace = Kokkos::DefaultExecutionSpace::memory_space;
    using ExecutionSpace = Kokkos::DefaultExecutionSpace;
    using DeviceType = Kokkos::Device<ExecutionSpace, MemorySpace>;
    
    double grid_delta[2] = {1.0, 1.0};
    double grid_min[2] = {0.0, 0.0};
    double grid_max[2] = {2.0, 2.0};
    
    double min_x = 0.0, min_y = 0.0, max_x = 2.0, max_y = 2.0;
    double dx = 1.0, dy = 1.0;
    double fat_radius = dx/10.0;

    Cajete::BrickGrid2D<double> global_grid(min_x, min_y, max_x, max_y, dx, dy);    

    Cajete::CellComplex2D<DeviceType> cell_complex(global_grid, fat_radius);
     
    REQUIRE( cell_complex._size == 15 );
    REQUIRE( cell_complex.graph_d.size() == 15 );
    REQUIRE( cell_complex.graph_h.size() == 15 );
    
    REQUIRE( cell_complex._offsets_h(0) == 0 );
    REQUIRE( cell_complex._offsets_h(1) == 5 );
    REQUIRE( cell_complex._offsets_h(2) == 12 );


    cell_complex.show();
}
