#include <Cabana_Core.hpp>
#include <string>
#include <iostream>

//Pick the memory space, choose UVM if cuda is defined(for convenience)
#ifdef KOKKOS_ENABLE_CUDA
using MemorySpace = Kokkos::CudaUVMSpace;
using ExecutionSpace = Kokkos::Cuda;
using DeviceType = Kokkos::Device<ExecutionSpace, MemorySpace>;
#else
using MemorySpace = Kokkos::HostSpace;
using ExecutionSpace = Kokkos::DefaultHostExecutionSpace;
using DeviceType = Kokkos::Device<ExecutionSpace, MemorySpace>;
#endif

namespace CjtEx {
template <class DeviceType, long int VectorLength = 4>
struct Graph {

    //We have something like follows for our sample node types:
    //  bool -> mark bit (byte in this case)
    //  int  -> node type
    //  size_t -> ptr to the edge block
    //  size_t -> number of edges this node has
    using NodeDataTypes = Cabana::MemberTypes<bool, int, size_t, size_t>;

    //We have something like follows for our sample edge types:
    // bool -> mark bit (byte in this case)
    // size_t -> Node pointed to
    using EdgeDataTypes = Cabana::MemberTypes<bool, size_t>;

    //choose the number of tuples for the aosoa i.e. number of our DataTypes
    size_t num_node_tuple;
    size_t num_edge_tuple;
   
    //information about the number of pools we're gonna have
    int num_node_pools;
    int num_edge_pools;

    //Set up the pool type
    using pool_type = Kokkos::View<size_t*, DeviceType>; 
   
    //The views for the pool capacities
    pool_type node_pool_capacity;
    pool_type edge_pool_capacity;

    //Put the pool offsets into a kokkos view of offsets
    pool_type node_pool_offsets;
    pool_type edge_pool_offsets;
    
    //set up the pool sizes
    pool_type node_pool_sizes;
    pool_type edge_pool_sizes;

    //The required AoSoA types for the graph
    Cabana::AoSoA<NodeDataTypes, DeviceType, VectorLength> nodes;
    Cabana::AoSoA<EdgeDataTypes, DeviceType, VectorLength> edges;

    //Default construction fo the graph sizes
    Graph(size_t _num_node_tuple, size_t _num_edge_tuple, int _num_pools)
        : num_node_tuple(_num_node_tuple) 
        , num_edge_tuple(_num_edge_tuple)
        , num_node_pools(_num_pools)
        , num_edge_pools(_num_pools)
        , nodes("my_nodes", _num_node_tuple)
        , edges("my_edges", _num_edge_tuple)
        , node_pool_capacity("NodePoolCapacity", _num_pools)
        , edge_pool_capacity("EdgePoolCapacity", _num_pools)
        , node_pool_offsets("NodePoolOffsets", _num_pools)
        , edge_pool_offsets("EdgePoolOffsets", _num_pools)
        , node_pool_sizes("NodePoolSizes", _num_pools)
        , edge_pool_sizes("EdgePoolSizes", _num_pools)
    {
        //
        //call the slicer here!
        //

        //Set on the device(we can do this thanks to UVM)
        //setting the node pools
        for(auto i = 0; i < num_node_pools; i++) {
            //we set the last pool to be slightly larger if ther is a remainder
            auto remainder = num_node_tuple % num_node_pools;
            auto round_down = num_node_tuple - remainder;
            auto divided = round_down / num_node_pools;
            auto capacity = i < num_node_pools - 1 ? divided : divided + remainder;
            node_pool_capacity( i ) = capacity;

        }

        //setting the edge pools
        for(auto i = 0; i < num_edge_pools; i++) {
            //we set the last pool to be slightly larger if ther is a remainder
            auto remainder = num_edge_tuple % num_edge_pools;
            auto round_down = num_edge_tuple - remainder;
            auto divided = round_down / num_edge_pools;
            auto capacity = i < num_edge_pools - 1 ? divided : divided + remainder;
            edge_pool_capacity( i ) = capacity;

        }

        //set the offsets with a simple serial scan
        auto node_sum = 0; auto edge_sum = 0;
        for(auto i = 0; i < num_node_pools; i++) {
            node_pool_offsets(i) = node_sum;
            edge_pool_offsets(i) = edge_sum;
            
            node_sum += node_pool_capacity(i);
            edge_sum += edge_pool_capacity(i);
        }
    
        //Set sized to zero via a deep copy
        Kokkos::deep_copy(node_pool_sizes, 0);
        Kokkos::deep_copy(edge_pool_sizes, 0);
    }

    
    KOKKOS_INLINE_FUNCTION
    long int node_allocate(int p, size_t n) const {
        
        //We need to atomically ask the node pool for some space,
        auto c = Kokkos::atomic_fetch_add(&node_pool_sizes(p), n);

        //Check to see if we are out of node pool capacity
        if(c+n >= node_pool_capacity(p)) {
            printf("OOM err in node pool %d, allocation failed!\n", p);
            Kokkos::atomic_sub(&node_pool_sizes(p), n);
            return -1;
        } else {
            return node_pool_offsets(p) + c; //Global node index for insertion
        }
    }

    KOKKOS_INLINE_FUNCTION
    long int edge_allocate(int p, size_t n) const {
        
        //We need to atomically ask the edge pool for some space
        auto c = Kokkos::atomic_fetch_add(&edge_pool_sizes(p), n);

        //Check to see if we are out of edge pool capacity
        if(c+n >= edge_pool_capacity(p)) {
            printf("OOM err in edge pool %d, allocation failed!\n", p);
            Kokkos::atomic_sub(&edge_pool_sizes(p), n);
            return -1;
        } else {
            return edge_pool_offsets(p) + c; //Global edge index for insertion
        }
    }


    KOKKOS_INLINE_FUNCTION
    void insert(size_t idx) {
        //Insert the node and mark it as not empty, set the type
        //mark_slice(j) = true;
        //type_slice(j) = p; //for now just setting the type to the pool it's in
        //ptr_slice(j) = 0;
        //num_edge_slice(j) = 0;
        //
        //if we wanted to add edges here, we'd also need to atomically ask for them
        //from an edge pool!
    }

    //function to print information about the graph
    void print_stats() { 
        std::cout << std::endl;
        for(auto i = 0; i < 80; i++)
            std::cout << "-";
        std::cout << std::endl;
        
        //Print out the size and capacity, we expect capacity to be slightly larger if num_tuple is
        //not divisible by vector lenght
        std::cout << "Nodes [Size, Capacity, NumSOA]: " << nodes.size() 
            << " " << nodes.capacity() << " " << nodes.numSoA() <<  "\n";
        std::cout << "Nodes [Size, Capacity, NumSoA]: " << edges.size() 
            << " " << edges.capacity() << " " << edges.numSoA() << "\n";
    
        /*
        //print the node pool capacities
        print1D(node_pool_capacity, "Node Pool Capacities");

        //print the edge pool capacities
        print1D(edge_pool_capacity, "Edge Pool Capacities");

        //print the node pool offsets
        print1D(node_pool_offsets, "Node Pool Offsets");

        //print the edge pool offsets
        print1D(edge_pool_offsets, "Edge Pool Offsets");
        
        //print the node pool sizes
        print1D(node_pool_sizes, "Node Pool Sizes");
    
        //print the edge pool sizes
        print1D(edge_pool_sizes, "Edge Pool Sizes");
        */
        for(auto i = 0; i < 80; i++)
            std::cout << "-";
        std::cout << std::endl;
    }

    template <class T>
    void print1D(T& item, std::string msg) {
        std::cout << msg << ": ";
        for(auto i = 0; i < item.size(); i++) {
            std::cout << item(i) << " ";
        } std::cout << std::endl;
    }
};

};

int main(int argc, char * argv[]) {
    Kokkos::ScopeGuard scope_guard(argc, argv);

    //choose the number of tuples for the aosoa i.e. number of our DataTypes
    int num_node_tuple = 80'000'000;//31;
    int num_edge_tuple = 240'000'000;//*90; //We expect more edges than tuples

    //Now we must determine how many memory pools the node array and the edge array must have
    //We could simply set it to one, but in general it should be no larger thant the number of nodes
    int num_pools = 6'000'000;//3; //note num node pools is identical to num edge pools for now
   
    //Vector length for the SoAs
    const int VectorLength = 4; //length depends on the system arch
    
    //Declare our graph with the give sizes, for now we have the node types harcoded
    CjtEx::Graph<DeviceType, VectorLength> graph(num_node_tuple, num_edge_tuple, num_pools);
    
    //Print intial stats
    graph.print_stats();
    //std::cin.get();
    //slice the node and edge marks
    auto node_mark_slice = Cabana::slice<0>(graph.nodes);
    auto edge_mark_slice = Cabana::slice<0>(graph.edges);
    
    Cabana::deep_copy(node_mark_slice, false); //ensures all nodes are marked as empty
    Cabana::deep_copy(edge_mark_slice, false); //ensures all edges are marked as empty

    //slice the rest of the node attributes
    auto type_slice = Cabana::slice<1>(graph.nodes);
    auto ptr_slice =  Cabana::slice<2>(graph.nodes);
    auto num_edge_slice = Cabana::slice<3>(graph.nodes);
    
    //slice the edge connection attribute
    auto con_slice = Cabana::slice<1>(graph.edges);

    //Decided the number of insertions for the test
    auto num_node_insertions = graph.nodes.size() - 8'000'000; //10;
    auto num_edges_per_node = 3;
    
    //start the timer
    Kokkos::Timer timer;
    
    //without a range policy it defaults execution the default memory space from compilation
    Kokkos::parallel_for("Pool Based Node Insertion", num_node_insertions, KOKKOS_LAMBDA(const int i) {
        
        int p = i % graph.num_node_pools; //Our insertion pools is evenly spread accross the threads
        int n = 1; //number of nodes we ask for allocation for
        
        auto j = graph.node_allocate(p, n);
        
        if(j >= 0) { //successful allocation request
            //Insert the node and mark it as not empty, set the type
            node_mark_slice(j) = true;
            type_slice(j) = p; //for now just setting the type to the pool it's in
            
            //if we wanted to add edges here, we'd also need to atomically ask for them
            //from an edge pool!
            auto k = graph.edge_allocate(p, num_edges_per_node);
            if(k >= 0) {//successful edge allocation
                for(auto i = 0; i < num_edges_per_node; i++) {    
                    edge_mark_slice(k+i) = true;
                    con_slice(k+i) = j; //self connection in this test
                }
                ptr_slice(j) = k;
                num_edge_slice(j) = num_edges_per_node;
            }
        }
    });
    Kokkos::fence();
    
    auto time = timer.seconds();
    timer.reset();
    std::cout << "Test took " << time << " seconds...\n";

    //now to check for correctness, lest loop through and count all the node marked bits
    auto node_mark_count = 0;
    for(auto i = 0; i < node_mark_slice.size(); i++) {
        if(node_mark_slice(i))
            node_mark_count++;
    }
    std::cout << "Asked for " << num_node_insertions << " insertions, and found " 
        << node_mark_count << " marks!\n";

    
    //now to check for correctness, lest loop through and count all the edge marked bits
    auto edge_mark_count = 0;
    for(auto i = 0; i < edge_mark_slice.size(); i++) {
        if(edge_mark_slice(i))
            edge_mark_count++;
    }
    std::cout << "Asked for " << num_node_insertions*num_edges_per_node << " insertions, and found " 
        << edge_mark_count << " marks!\n";


    //print final stats
    graph.print_stats();

    return 0;

}
