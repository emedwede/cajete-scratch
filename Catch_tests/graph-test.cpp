#include "catch.hpp"
#include "graph.hpp"
#include "mt_inits.hpp"
#include "visualization.hpp"
#include <set>
#include <utility>

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
}

//We do the shuffle test by checking for referential integrity
//after the shuffle
TEST_CASE("Graph Shuffle Test", "[graph_test]") {
    
    Cajete::Graph<DeviceType> system(5);
    mt3_uniform_init(system, 35);
    
    //We use pair since (2, 1) != (1, 2) i.e. unique
    using PairType = std::pair<long int, long int>;
    std::set<PairType> setA;
    std::set<PairType> setB;

    for(auto i = 0; i < system.get_size(); i++) {
        auto src_id = system.id_h(i);
        for(int j = 0; j < 4; j++) {
            auto con = system.edges_h(i, j);
            if(con != -1) {
                auto dest_id = system.id_h(con);
                setA.insert(std::make_pair(src_id, dest_id));
            }
        }
    }
   
    Visualizer<DeviceType> writer;
    writer.write_vis(system, "test_vis_step_1");
    //shuffle several times to ensure we have no errors
    for(auto i = 0; i < 10; i++)
        system.shuffle();
    writer.write_vis(system, "test_vis_step_2");

    for(auto i = 0; i < system.get_size(); i++) {
        auto src_id = system.id_h(i);
        for(int j = 0; j < 4; j++) {
            auto con = system.edges_h(i, j);
            if(con != -1) {
                auto dest_id = system.id_h(con);
                setB.insert(std::make_pair(src_id, dest_id));
            }
        }
    }
    
    bool set_match = (setA == setB);
    REQUIRE( set_match == true );
}

TEST_CASE("Graph Copy Test", "[graph_test]") {
    //Make graph of size 11 nodes
    Cajete::Graph<DeviceType> graph_current(11);
    
    //init a graph with 3 Microtubules
    mt3_uniform_init(graph_current, 3);
    
    //Make an empty graph, since copy will resize it
    Cajete::Graph<DeviceType> graph_old(0);
    
    //Copy current graph to old graph
    Cajete::graph_copy(graph_old, graph_current);

    //First check that the copy worked by picking an arbitrary position
    REQUIRE( graph_old.positions_h(2, 0) == graph_current.positions_h(2, 0) );
    
    //shuffle current graph
    graph_current.shuffle();

    //Check that they are in fact deep copies, since a shuffle would effect 
    //a shallow copy
    REQUIRE( graph_old.positions_h(2, 0) != graph_current.positions_h(2, 0) );

}

