#include "catch.hpp"
#include "grid.hpp"

TEST_CASE( "2D Brick Grid Test", "[grid_test]" )
{
    double min_x = 0.0, min_y = 0.0, max_x = 2.0, max_y = 2.0;
    double dx = 1.0, dy = 1.0;

    Cajete::BrickGrid2D<double> global_grid(min_x, min_y, max_x, max_y, dx, dy);    
    
    REQUIRE(global_grid.totalNumCells() == 5);

    Cajete::CartesianGrid2D<double> local_grid;
    local_grid.init(min_x, min_y, max_x, max_y, dx, dy);
    
    REQUIRE(local_grid.totalNumCells() == 4);
}

