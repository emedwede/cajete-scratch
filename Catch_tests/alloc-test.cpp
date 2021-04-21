#include "catch.hpp"
#include "graph_alloc.hpp"

#ifdef KOKKOS_ENABLE_CUDA
using DeviceType = Kokkos::CudaUVMSpace;
#else
using DeviceType = Kokkos::HostSpace;
#endif

TEST_CASE( "Alloc Test", "[alloc_test]" )
{
    Cajete::NodeList<DeviceType> node_list(2, 10, 5);
    
    REQUIRE( node_list.node_data.size() == 10 );

    for(auto i = 0; i < node_list.node_data.size(); i++) {
        node_list.print_at(i);
    }

    REQUIRE( node_list.num_edges.size() == 10 );

    REQUIRE( node_list.edge_ptr.size() == 10);

    Cajete::Allocator<Cajete::NodeList<DeviceType>, DeviceType> alloc(node_list, 1);
    
    REQUIRE( alloc.handle.node_data.size() == 10 );

}
