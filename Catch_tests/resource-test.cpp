#include "catch.hpp"

#include "aosoa_resource.hpp"
#include "binner.hpp"

#ifdef KOKKOS_ENABLE_CUDA
using DeviceType = Kokkos::CudaUVMSpace;
#else
using DeviceType = Kokkos::HostSpace;
#endif
   
TEST_CASE( "Resource Init Test", "[resource_test]" )
{
    using NodeDataTypes = Cabana::MemberTypes<int, double, bool>;
    Cajete::AosoaResource<NodeDataTypes, DeviceType> nodes("NodeResource", 100);
    nodes.allocate(5, 1);
    nodes.deallocate(5, 1);

    REQUIRE(nodes.get_capacity() == 100);
    
    auto nodes_shallow_copy = nodes.get_resource();
    auto slice_a = Cabana::slice<0>(nodes.get_resource());
    slice_a(53) = 26;
    auto slice_b = Cabana::slice<0>(nodes_shallow_copy);
    REQUIRE(slice_a(53) == slice_b(53));
}

TEST_CASE( "Binner Test", "[binner_test]" )
{
    int nbins = 10;
    int bsize = 4;

    Cajete::BinningData<DeviceType> binner(nbins, bsize);
    
    REQUIRE(binner.numBin() == 10);
    REQUIRE(binner.binOffset(3) == 12);
    REQUIRE(binner.binCapacity(8) == 4);
    REQUIRE(binner.binSize(8) == 0);

    auto binner_copy = binner;

    REQUIRE(binner_copy.binOffset(4) == 16);
}


