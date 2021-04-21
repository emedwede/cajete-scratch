#ifndef __CAJETE_GRAPH_ALLOC_HPP
#define __CAJETE_GRAPH_ALLOC_HPP

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_DualView.hpp>
#include <Kokkos_Bitset.hpp>

#include <string>
#include <iostream>

namespace Cajete {

//Each node has a set of associated features
//Each node has a block of memory pointing to some number pointers to edge blocks associated with it
//Each of those edge blocks need not be contiguous, but the the pointers to blocks must be
//all edges in a block must be contiguous
//all edge blocks but the last edge block must be full

//Important consideration must be taken when differentiating between
//parallel batch insertions and parallel concurrent threads attempting to
//do localized insertions at different rates
//batch insertions allow us to quickly approximate our memory needs based on the batch
//asynchronous insertions on the other hand mean that we may need pools per thread
template<class DeviceType>
struct NodeList {
    
    struct NodeLabel {
        enum Fields {
            Free, //Is the node block free, [bool]
            Type, //Type of the node        [unsigned short int] (likely)
            EdgePtr, //Edge Block Start  [unsigned       int]
            Nbrs,  //Number of Edges     [unsigned       int]
        };
    };

    using NodeDataTypes = Cabana::MemberTypes<bool, unsigned short int, size_t, size_t>;

    using view_type = Kokkos::View<size_t*, DeviceType>;
    using prim_type = Kokkos::View<size_t, DeviceType>;
    using NodeListType = Cabana::AoSoA<NodeDataTypes, DeviceType>;

    using bit_status_t = typename NodeListType::template member_slice_type<0>;
    using node_type_t  = typename NodeListType::template member_slice_type<1>; 
    using edge_ptr_t = typename NodeListType::template member_slice_type<2>;
    using num_edges_t = typename NodeListType::template member_slice_type<3>;

    NodeListType node_data;
    view_type edge_blocks;
   
    bit_status_t bit_status;
    node_type_t node_type;
    edge_ptr_t edge_ptr;
    num_edges_t num_edges;

    prim_type num_pools;
    view_type pool_capacity;
    view_type pool_size;

    //Need edge_blocks for at least as many nodes
    NodeList(size_t _capacity, size_t _block_size, size_t _num_pools = 1) 
        : node_data("Node Data", _capacity*_num_pools)
        , edge_blocks("Edge Blocks", _block_size*_capacity*_num_pools)
        , pool_capacity("Capacity of Pools", _num_pools)
        , pool_size("Sizes of Pools", _num_pools)
    {
        Kokkos::deep_copy(pool_size, 0);
        
        //Assumes all pools are the same size
        Kokkos::deep_copy(pool_capacity, _capacity);     
        
        //slice the node aosoa
        slice();
        init();
    }
    
    void init () {
        Cabana::deep_copy(bit_status, 0);
        Cabana::deep_copy(node_type, 0);
        Cabana::deep_copy(edge_ptr, 0);
        Cabana::deep_copy(num_edges, 0);
    }

    void print_at (size_t idx) {
        slice();
        std::cout << "\n--------------------------------\n";
        std::cout << "Data at index " << idx << "\n";
        std::cout << "++++++++++++++++++++++++++++++++\n";
        std::cout << "\tBit Status: " << bit_status(idx) << "\n";
        std::cout << "\tNode Type : " << node_type(idx) << "\n";
        std::cout << "\tEdge Ptr  : " << edge_ptr(idx) << "\n";
        std::cout << "\tNum Edges : " << num_edges(idx) << "\n";
        std::cout << "--------------------------------\n";
    }

    void slice () {
        bit_status = Cabana::slice<0>(node_data);
        node_type = Cabana::slice<1>(node_data);
        edge_ptr = Cabana::slice<2>(node_data);
        num_edges = Cabana::slice<3>(node_data);
    }

    //returns index of allocation of n slots in pool p
    KOKKOS_INLINE_FUNCTION
    long int allocate(size_t n, size_t p) const {
        long int c = Kokkos::atomic_fetch_add( &pool_size(p), n );
        if(c >= pool_capacity(p)) {
            printf("OOM err in pool %zu, allocation failed!\n", p);
            Kokkos::atomic_sub(&pool_size(p), n);//( &size(), n );
            return -1;
        } else {
            //TODO: need to return the offset scan of pool caps
            return ( pool_capacity(p) * p + c );         
        }
    }


};

/* A memory manager is in charge of an allocator and memory
 * compression and decompression. There are times when pools of
 * allocated memory may need to be consolidated. In general, the
 * implementation of this should be independent of the object 
 * requesting resource consolidation.
 */
struct MemoryManager {

};

//allocator takes an object pointer to a contiguous block of memory that has a function size defined
//upon first getting the resource the allocator is given a request for 
//a number of pools. The allocator must decide if that many pools are 
//possible for the given resources capacity. If not, the closest approximation is accomodated
//
//All allocators are assumed to point to empty memory when initialized
template <typename ResourceType, typename DeviceType>
struct Allocator {
    using view_type = Kokkos::View<size_t*, DeviceType>;
    using prim_type = Kokkos::View<size_t, DeviceType>;

    ResourceType handle;
    
    Allocator(ResourceType& _handle, size_t _num_pools) 
        : handle(_handle) 
        , pool_capacity("Capacity of Pools", _num_pools)
        , pool_size("Sizes of Pools", _num_pools)
    {
        auto capacity = _handle.node_data.size();
    }

    prim_type num_pools;
    view_type pool_capacity;
    view_type pool_size;


};

};

#endif
