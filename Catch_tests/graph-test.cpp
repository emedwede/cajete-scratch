#include "catch.hpp"
#include "graph.hpp"
#include "mt_inits.hpp"
#include "visualization.hpp"

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
    //Do a crude shuffle test
    system = Cajete::Graph<DeviceType>(5);
    mt3_uniform_init(system, 350);
    typename Cajete::Graph<DeviceType>::graph_t_h 
        temp_graph_h("Temp", system.reserve_size_h());
    Cabana::deep_copy(temp_graph_h, system.graph_h);
    auto temp_edges_h = Cabana::slice<2>(temp_graph_h);
    auto temp_id_h = Cabana::slice<3>(temp_graph_h);

    Visualizer<DeviceType> writer;
    writer.write_vis(system, "test_vis_step_1");
    //shuffle plenty of times to ensure we have no errors
    for(auto i = 0; i < 100; i++)
        system.shuffle();
    writer.write_vis(system, "test_vis_step_2");

    //TODO: Find a better way to write this
    for(auto i = 0; i < system.get_size(); i++) {
        int a = system.id_h(i);
        for(auto j = 0; j < 4; j++) {
            int con = system.edges_h(i, j);
            int b = -1;
            if(con != -1) {
                b = system.id_h(con);
                bool found = false;
                int a_t = temp_id_h(a);
                for(auto k = 0; k < 4; k++) {
                    if(temp_id_h(temp_edges_h(a, k)) == b)
                        found = true;
                }
                REQUIRE(found == true);
            }

        }
    }
}

