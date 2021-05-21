#include "catch.hpp"
#include <Kokkos_Core.hpp>

#include <Cajete_NodeTypes.hpp>
#include <Cajete_NodeList.hpp>

#include <iostream>

#ifdef KOKKOS_ENABLE_CUDA
using DeviceType = Kokkos::CudaUVMSpace;
#else
using DeviceType = Kokkos::HostSpace;
#endif
TEST_CASE( "Node List Test", "[node_list_test]" )
{
    REQUIRE(Cajete::Field::VertexType().label() == "vertex_type");
   
    using VertexType = Cajete::Field::VertexType;
    Cajete::Node<VertexType> my_tuple;

    auto i = Cajete::Node<VertexType>::traits::member_types::size;
    REQUIRE(i == 1);

    using list_type = Cajete::NodeList<DeviceType, VertexType>;

    list_type nodes("my_nodes");

    //------------------------------------------
    // Resize Test
    //------------------------------------------
    auto& aosoa = nodes.aosoa();
    std::size_t num_p = 15;
    aosoa.resize(num_p);

    REQUIRE(nodes.capacity() == num_p);

    
    //------------------------------------------
    // Slice Tests
    //------------------------------------------
    auto slice_0 = nodes.slice(VertexType());
    REQUIRE(slice_0.size() == num_p);
    REQUIRE(slice_0.label() == "vertex_type");

    //------------------------------------------
    // Get tests
    //------------------------------------------
    auto aosoa_host = Cabana::create_mirror_view_and_copy(Kokkos::HostSpace(), aosoa);
    Cabana::deep_copy(slice_0, 2);
    
    //Local modify
    Kokkos::parallel_for("modify", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, num_p),
            KOKKOS_LAMBDA(const int p) {
        //gets a copy of a node p
        auto node = nodes.get_node(p);
        
        //get the vertex of the tuple and set it
        get( node, VertexType() ) = p;
        
        //insert the node at point, overwrites what is at p!
        nodes.insert(p, node);
    });

    Cabana::deep_copy(aosoa_host, aosoa);
    auto host_slice_0 = Cabana::slice<0>(aosoa_host);
    for(auto p = 0; p < num_p; ++p) {
        REQUIRE( host_slice_0(p) == p );
    }
}

