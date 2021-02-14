#ifndef __CAJETE_GRAPH_HPP
#define __CAJETE_GRAPH_HPP

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>

namespace Cajete {

//All spatially embedded verticies need a minimum set of information
using NodeDataTypes =
    Cabana::MemberTypes<
        double[2], //position
        int, //type
        long int[4] //maximal number of edges
    >;
    
//Graphs are composed of nodes and edges
template<class DeviceType>
struct Graph {

    using graph_t_d = Cabana::AoSoA<NodeDataTypes, DeviceType>;
    using positions_t_d = typename graph_t_d::template member_slice_type<0>;
    using node_type_t_d = typename graph_t_d::template member_slice_type<1>;
    using edge_type_t_d = typename graph_t_d::template member_slice_type<2>;
    
    using graph_t_h = typename graph_t_d::host_mirror_type;
    using positions_t_h = typename graph_t_h::template member_slice_type<0>;
    using node_type_t_h = typename graph_t_h::template member_slice_type<1>;
    using edge_type_t_h = typename graph_t_h::template member_slice_type<2>;
  
    using count_t_d = Kokkos::View<size_t, typename DeviceType::execution_space>;
    using count_t_h = typename count_t_d::HostMirror;

    graph_t_d graph_d;
    graph_t_h graph_h;

    positions_t_d positions_d;
    positions_t_h positions_h;

    node_type_t_d nodes_d;
    node_type_t_h nodes_h;

    edge_type_t_d edges_d;
    edge_type_t_h edges_h;
    
    //current this defaults to default space
    count_t_d current_size_d;
    count_t_h current_size_h;

    count_t_d reserve_size_d;
    count_t_h reserve_size_h;

    float _r_f;

    //TODO: Add a reserve factor to help test pushes to a limit
    Graph(size_t n, float r_f=1.25)
        : graph_d("Graph on the Device", n)
        , graph_h("Graph on the Host", n)
        , positions_d(Cabana::slice<0>(graph_d))
        , positions_h(Cabana::slice<0>(graph_h))
        , nodes_d(Cabana::slice<1>(graph_d))
        , nodes_h(Cabana::slice<1>(graph_h))
        , edges_d(Cabana::slice<2>(graph_d))
        , edges_h(Cabana::slice<2>(graph_h))
        , current_size_d("Graph Size Device")
        , current_size_h("Graph Size Host")
        , reserve_size_d("Max Graph Size Device")
        , reserve_size_h("Max Graph Size Host")
        , _r_f(r_f)
    {
        if(_r_f <= 1.0)
            _r_f = 1.0;
        //deep copy allows us to set the value on the device easily
        Kokkos::deep_copy(current_size_d, n);
        Kokkos::deep_copy(current_size_h, n); //Excessive for the host
        reserve(_r_f);
        Cabana::deep_copy(positions_d, 0.0);
        Cabana::deep_copy(nodes_d, -1);
        Cabana::deep_copy(edges_d, -1);
        copy_device_to_host();
    }

    void reslice () {
        positions_d = Cabana::slice<0>(graph_d);
        positions_h = Cabana::slice<0>(graph_h);

        nodes_d = Cabana::slice<1>(graph_d);
        nodes_h = Cabana::slice<1>(graph_h);

        edges_d = Cabana::slice<2>(graph_d);
        edges_h = Cabana::slice<2>(graph_h);
    }
    
    //sets to default values on a range
    void set_default_values(size_t start, size_t end) {
       
        const auto& _positions_d = positions_d;
        const auto& _nodes_d = nodes_d;
        const auto& _edges_d = edges_d;
        //TODO: move to be provided via tagged dispatch    
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace> fill_policy(start, end);
        Kokkos::parallel_for("DefaultValueReserves", fill_policy, KOKKOS_LAMBDA(const int i) {
            _positions_d(i, 0) = 0.0; _positions_d(i, 1) = 0.0;
            for(auto j = 0; j < 4; j++) {
                _edges_d(i, j) = -1;
            }
            _nodes_d(i) = -1;
        });
        Kokkos::fence();
        copy_device_to_host();
        reslice();
    }

    //reserves some factor more of memory
    void reserve(float r_f=2.0) {
        if(r_f < 1.0)
            r_f = 2.0;
        reserve_size_h() = ceil(current_size_h()*r_f);
        Kokkos::deep_copy(reserve_size_d, reserve_size_h);
        //Here we need the underlying aosoa to have the same
        //size as reserve, so we can safely push to it on the
        //device without needing to reslice!
        //
        //Note: the true cabana reserve capacity is always >=
        graph_d.reserve(reserve_size_h());
        graph_d.resize(reserve_size_h());
        graph_h.reserve(reserve_size_h());
        graph_h.resize(reserve_size_h());
        reslice();
        set_default_values(current_size_h(), reserve_size_h());
    }

    //resizes our graph
    void resize_HostSafe(size_t n) {
        if(n > reserve_size_h()) {
            auto low = current_size_h();
            //reserves a little more memory
            current_size_h() = n;
            reserve(1.25);
            set_default_values(low, n); //TODO: fix this crazy logic
        } else if(n >= 0) {
            current_size_h() = n;
        } else {
            //do nothing
        }
        Kokkos::deep_copy(current_size_d, current_size_h);
        Kokkos::deep_copy(reserve_size_d, reserve_size_h);
    }

    //this allows us to resize up to our reserve 
    //on the device
    KOKKOS_INLINE_FUNCTION
    void resize_DeviceSafe(size_t n) const {
       //TODO: needs to let us push to our reserve
       //      and throw an error if we exceed?
    }

    void copy_host_to_device() {
        Cabana::deep_copy(graph_d, graph_h);
        reslice();
    }

    void copy_device_to_host() {
        Cabana::deep_copy(graph_h, graph_d);
        reslice();
    }

    size_t get_size() {
        Kokkos::deep_copy(current_size_h, current_size_d);
        return current_size_h();
    }

    //graph aosoa capacity may be a tad higher
    size_t capacity() { 
        return reserve_size_h();
    }
    
    //graph needs insertions 
    //
    //Pushes a point to the end of our graph atomically
    KOKKOS_INLINE_FUNCTION
    void atomic_push() const { //Not a bad choice if we have infrequent parallel pushes
        size_t id = Kokkos::atomic_fetch_add(&current_size_d(), 1); 
    }

    //prints out everything, could get pretty crazy, please refer to spiderman quote
    void show() {
        for(auto i = 0; i < current_size_h(); i++) {
            std::cout << "Node: " << i << "\n";
            std::cout << "Positions: [ " << positions_h(i, 0) << " , " << positions_h(i, 1) << " ]\n";
            std::cout << "Node Type: [ " << nodes_h(i) << " ]\n";
            std::cout << "Edge Link: [";
            for(auto j = 0; j < 4; j++) {
                std::cout << " " << edges_h(i, j) << " ";
            } std::cout << "]\n\n";
        }
    }

    //graph needs removals
    
    //graph needs customizable rewrites
    
    //we want insertions, removals and rewrites to be safely concurrent with respect to 
    //spatial locality

    //graph needs to store edges and be able to accomodate new edges up to a maximal degree

    //graph needs to be sortable and shufflable
};

/***************************************************
 *Extra Notes:
 *
 *the cell complex and the cell list are what allow 
 *us to apply sorting constraints and do localized
 *graph searches like breadth first etc.
 *
 *Pros and cons for edge and node decoupling:
 *If we have a seperate edge list it make be easier
 *to add and remove edges in a more memory efficient way.
 *But, we would do this at the cost of memory locality and
 *algorithmic simplicity. 
 *
 * Things I want to do:
 * step (1) construct a simple graph
 * step (2) outp
 * step (2) randomly shuffle the graph nodes and 
 *          preserve edge references.
 * step (3) 
 ***************************************************/
};

#endif
