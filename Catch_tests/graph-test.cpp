#include "catch.hpp"
#include "graph.hpp"

using MemorySpace = Kokkos::DefaultExecutionSpace::memory_space;
using ExecutionSpace = Kokkos::DefaultExecutionSpace;
using DeviceType = Kokkos::Device<ExecutionSpace, MemorySpace>;
   
TEST_CASE( "Graph Initialization Test", "[graph_test]" )
{
    size_t size = 10;
    Cajete::Graph<DeviceType> system(size);
    
    //If these copies fail, so will the test
    system.copy_host_to_device();
    system.copy_device_to_host();

    REQUIRE( size == system.graph_d.size() );
    REQUIRE( size <= system.capacity() );
    REQUIRE( size == system.get_size() );
    size_t num_pushes = 5000;
    
    Kokkos::parallel_for(num_pushes, KOKKOS_LAMBDA(const int i) {
        system.atomic_push();
    });

    REQUIRE( ( size + num_pushes ) == system.get_size() );
}

