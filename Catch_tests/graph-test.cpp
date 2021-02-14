#include "catch.hpp"
#include "graph.hpp"
#include "mt_inits.hpp"

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
    

    REQUIRE( size == system.get_size() );
    REQUIRE( system.capacity() <= system.graph_d.size() );
    REQUIRE( system.get_size() <= system.capacity() );
    REQUIRE( system.capacity() <= system.graph_d.capacity());
    
    float reserve_factor = 3.2;
    size = 20;
    system.reserve(reserve_factor);
    system.resize_HostSafe(size);
    REQUIRE( size == system.get_size() );
    REQUIRE( system.capacity() <= system.graph_d.size() );
    REQUIRE( system.get_size() <= system.capacity() );
    REQUIRE( system.capacity() <= system.graph_d.capacity());
    
    size = 54;
    system.resize_HostSafe(size);
    REQUIRE( size == system.get_size() );
    REQUIRE( system.capacity() <= system.graph_d.size() );
    REQUIRE( system.get_size() <= system.capacity() );
    REQUIRE( system.capacity() <= system.graph_d.capacity());

    size_t num_pushes = 5000;
    
    Kokkos::parallel_for(num_pushes, KOKKOS_LAMBDA(const int i) {
        system.atomic_push();
    });

    REQUIRE( ( size + num_pushes ) == system.get_size() );
    
    //system = Cajete::Graph<DeviceType>(5);
    //system.show();
    //mt3_uniform_init(system, 3);
    //system.show();
}

