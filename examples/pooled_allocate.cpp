#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_DualView.hpp>
#include <Kokkos_Bitset.hpp>

#include <string>
#include <iostream>

//using view_type = Kokkos::DualView<int*, Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace>;

struct NodeLabel {
    enum Fields {
        Free, //Is the node block free, [bool]
        Type, //Type of the node        [unsigned short int] (likely)
        BlockStart, //Edge Block Start  [unsigned       int]
        BlockEnd,  //Edge Block End     [unsigned       int]
    };
};

//Bool: marked flag, 1 for full, 0 for empty

using NodeDataTypes = Cabana::MemberTypes<bool, unsigned short int, size_t, size_t>;

//Each node has a set of associated features
//Each node has a block of memory pointing to some number pointers to edge blocks associated with it
//Each of those edge blocks need not be contiguous, but the the pointers to blocks must be
//all edges in a block must be contiguous
//all edge blocks but the last edge block must be full

#ifdef KOKKOS_ENABLE_CUDA
using view_type = Kokkos::View<size_t*, Kokkos::CudaUVMSpace>;
using prim_type = Kokkos::View<size_t, Kokkos::CudaUVMSpace>;
using NodeListType = Cabana::AoSoA<NodeDataTypes, Kokkos::CudaUVMSpace>;
#else
using view_type = Kokkos::View<size_t*, Kokkos::HostSpace>;
using prim_type = Kokkos::View<size_t, Kokkos::HostSpace>;
using NodeListType = Cabana::AoSoA<NodeDataTypes, Kokkos::HostSpace>;
#endif

using bit_status_t = typename NodeListType::template member_slice_type<0>;
using node_type_t  = typename NodeListType::template member_slice_type<1>; 
using block_start_t = typename NodeListType::template member_slice_type<2>;
using block_end_t = typename NodeListType::template member_slice_type<3>;

//using view_type2 = Kokkos::View<std::byte*, Kokkos::HostSpace>;

struct NodeList {
    NodeListType node_data;
    view_type edge_blocks;
   
    bit_status_t bit_status;
    node_type_t node_type;
    block_start_t block_start;
    block_end_t block_end;

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
    }

    void slice () {
        bit_status = Cabana::slice<0>(node_data);
        node_type = Cabana::slice<1>(node_data);
        block_start = Cabana::slice<2>(node_data);
        block_end = Cabana::slice<3>(node_data);
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

template<typename ResourceType>
struct kokkos_alloc {
    
    kokkos_alloc(ResourceType _resource) {}
};

//how would I start a graph/node list from scratch and allow for parallel 
//insertions? Simple idea: have some expectation for the number of insertions
//and design size the pools accordingly!
struct meta_heap {
    view_type data;

    meta_heap(size_t _capacity, size_t _num_pools = 1) 
        : data("MetaHeapData", _capacity*_num_pools)
        , size("HeapSize")
        , capacity("Capacity")
        , num_pools("Number of Pools")
        , pool_size("Pool Sizes", _num_pools)
        , pool_capacity("Pool Capacities", _num_pools)
    {
        size() = 0;
        capacity() = _capacity;
        num_pools() = _num_pools;
        Kokkos::deep_copy(pool_size, 0);
        for(auto i = 0; i < _num_pools; i++) {
            pool_capacity(i) = _capacity; //fix this
        }
    }

    //returns index of allocation of n slots
    KOKKOS_INLINE_FUNCTION
    long int allocate(size_t n) const {
        long int c = Kokkos::atomic_fetch_add( &size(), n );
        if(c >= capacity()) {
            printf("Out of memory error, device allocation failed!\n");
            Kokkos::atomic_sub(&size(), n);//( &size(), n );
            return -1;
        } else {
            return ( c );
        }
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
            return ( capacity() * p + c );         
        }
    }

    void print_pool_stats() {
        std::cout << "[Number of Pools : [Size, Capacity]]: \n";
        std::cout << "\tNumber of Pools == " << pool_capacity.size() << "\n";
        for(auto i = 0; i < pool_capacity.size(); i++) {
            std::cout << "\t\t[ " << pool_size(i) << " , "
                << pool_capacity(i) << " ] \n";
        }
    }

    void print_stats() {
        std::cout << "[Size, Capacity, Pools]: " << "[ " 
            << size() << " , "
            << capacity() << " , " 
            << num_pools() << " ]\n";
    }
    prim_type capacity;
    prim_type size;
    prim_type num_pools;
    view_type pool_capacity;
    view_type pool_size;
};

int main(int argc, char* argv[]) {
   
    Kokkos::ScopeGuard scope_guard(argc, argv);
    NodeListType aosoa("MyNodeList", 100);
    
    NodeList nodes(100, 10, 5);
    std::cout << "Bit Status Size: " << nodes.bit_status.size() << std::endl;
    std::cout << "Running pooled allocate...\n";    
    //size_t size = 10'000'000;
    size_t size_par = 62'500;
    size_t num_pool = 3200;
    size_t size = size_par*num_pool;
    std::cout << sizeof(size_t) << std::endl;
    meta_heap arr(size);
    meta_heap arr_par(size_par, num_pool);
    //arr.print_stats();
    //arr_par.print_pool_stats();
    
    //std::cin.get();
    Kokkos::Timer timer;

    Kokkos::parallel_for("Thread Per Pool Push Test", 
    num_pool, KOKKOS_LAMBDA(const int i) {
        //for each element in the pool push
        for(auto j = 0; j < size_par; j++) {
            //Get the allocation index for a block size 1
            int idx = arr_par.allocate(1, i);
            //assign to the allocation space
            arr_par.data(idx) = idx;
        }
    });

    Kokkos::fence();
    auto time = timer.seconds();
    timer.reset();
    //arr_par.print_pool_stats();

    std::cout << "Test took " << time << " seconds...\n";
    
    //Kokkos::Bitset<Kokkos::HostSpace> bitset(100u);
    //std::cout << "BitSet Size: " << bitset.size() << std::endl;
    //bitset.set(10);
    //bitset.set(3);
    //bitset.reset(10);
    //std::cout << "BitSet Count: " << bitset.count() << std::endl;

    //arr.print_stats();
    /*std::cout << "Contents: ";
    for(auto i = 0; i < size_par*num_pool; i++) {
        std::cout << arr_par.data(i) << " ";
    } std::cout << std::endl;*/
    return 0;
}
