#include "catch.hpp"
#include "grid.hpp"
#include "particle_array.hpp"
#include "cell_complex.hpp"
#include "ll_cell_list.hpp"

TEST_CASE("List of Linked Cell List Test" "[cell_list_test]") 
{
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

    REQUIRE(cell_complex._global_grid.totalNumCells() == 5);
    
    REQUIRE(cell_complex.grid_view_h(0).totalNumCells() == 4);

}
