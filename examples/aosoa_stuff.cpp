#include <Cabana_Core.hpp>

#include <iostream>

int main(int argc, char * argv[]) {
    Kokkos::ScopeGuard scope_guard(argc, argv);

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

    //Vector length for the SoAs
    const int VectorLength = 4; //length depeends on the system arch

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

    //choose the number of tuples for the aosoa i.e. number of our DataTypes
    int num_node_tuple = 31;
    int num_edge_tuple = 90; //We expect more edges than tuples

    //Declare our AoSoAs for the graph
    Cabana::AoSoA<NodeDataTypes, DeviceType, VectorLength> nodes("my_nodes", num_node_tuple);
    Cabana::AoSoA<EdgeDataTypes, DeviceType, VectorLength> edges("my_edges", num_edge_tuple);

    //Print out the size and capacity, we expect capacity to be slightly larger if num_tuple is
    //not divisible by vector lenght
    std::cout << "Nodes [Size, Capacity, NumSOA]: " << nodes.size() 
        << " " << nodes.capacity() << " " << nodes.numSoA() <<  "\n";
    std::cout << "Nodes [Size, Capacity, NumSoA]: " << edges.size() 
        << " " << edges.capacity() << " " << edges.numSoA() << "\n";
   
    //Now we must determine how many memory pools the node array and the edge array must have
    //We could simply set it to one, but in general it should be no larger thant the number of nodes
    int num_node_pools = 3;
    int num_edge_pools = num_node_pools; //We want the pools for nodes to correspond to those for edges

    //Set up the pool type
    using pool_type = Kokkos::View<size_t*, DeviceType>; 
   
    //set up the initial pool capacities
    pool_type node_pool_capacity("NodePoolCapacity", num_node_pools);
    pool_type edge_pool_capacity("EdgePoolCapacity", num_edge_pools);

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
    
    //Set on the device(we can do this thanks to UVM)
    //setting the edge pools
    for(auto i = 0; i < num_edge_pools; i++) {
        //we set the last pool to be slightly larger if ther is a remainder
        auto remainder = num_edge_tuple % num_edge_pools;
        auto round_down = num_edge_tuple - remainder;
        auto divided = round_down / num_edge_pools;
        auto capacity = i < num_edge_pools - 1 ? divided : divided + remainder;
        edge_pool_capacity( i ) = capacity;
    }

    //print the node pool capacities
    std::cout << "Node Pool Capacities: ";
    for(auto i = 0; i < num_node_pools; i++) {
        std::cout << node_pool_capacity(i) << " ";
    } std::cout << std::endl;

    //print the edge pool capacities
    std::cout << "Edge Pool Capacities: ";
    for(auto i = 0; i < num_node_pools; i++) {
        std::cout << edge_pool_capacity(i) << " ";
    } std::cout << std::endl;

    //Put the pool offsets into a kokkos view of offsets
    pool_type node_pool_offsets("NodePoolOffsets", num_node_pools);
    pool_type edge_pool_offsets("EdgePoolOffsets", num_edge_pools);
    
    //set the offsets with a simple serial scan
    auto node_sum = 0; auto edge_sum = 0;
    for(auto i = 0; i < num_node_pools; i++) {
        node_pool_offsets(i) = node_sum;
        edge_pool_offsets(i) = edge_sum;
        
        node_sum += node_pool_capacity(i);
        edge_sum += edge_pool_capacity(i);
    }
    
    //print the node pool offsets
    std::cout << "Node Pool Offsets: ";
    for(auto i = 0; i < num_node_pools; i++) {
        std::cout << node_pool_offsets(i) << " ";
    } std::cout << std::endl;

    //print the edge pool offsets
    std::cout << "Edge Pool Offsets: ";
    for(auto i = 0; i < num_edge_pools; i++) {
        std::cout << edge_pool_offsets(i) << " ";
    } std::cout << std::endl;

    //set up the pool sizes
    pool_type node_pool_sizes("NodePoolSizes", num_node_pools);
    pool_type edge_pool_sizes("EdgePoolSizes", num_edge_pools);

    //Set them to zero via a deep copy
    Kokkos::deep_copy(node_pool_sizes, 0);
    Kokkos::deep_copy(edge_pool_sizes, 0);

    auto mark_slice = Cabana::slice<0>(nodes);
    Cabana::deep_copy(mark_slice, false); //ensures all nodes are marked as empty
    for(auto i = 0; i < num_node_tuple; i++) {
        std::cout << mark_slice(i) << " ";
    } std::cout << std::endl;

    //without a range policy it defaults execution the defualt memory space from compilation
    Kokkos::parallel_for("Pool Based Node Insertion", num_node_pools, KOKKOS_LAMBDA(const int i) {
        //here is where we need to implement some sort of atomic insertions
        //auto pool_number
    });

    return 0;

}
