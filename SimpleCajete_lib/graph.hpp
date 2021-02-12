#ifndef __CAJETE_GRAPH_HPP
#define __CAJETE_GRAPH_HPP

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>

namespace Cajete {

//All spatially embedded verticies need a minimum set of information
using NodeDataTypes =
    Cabana::MemberTypes<
        double[2], //position
        size_t, //type
        size_t[4] //maximal number of edges
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
  
    using count_t_d = Kokkos::View<size_t>;

    graph_t_d graph_d;
    graph_t_h graph_h;

    positions_t_d positions_d;
    positions_t_h positions_h;

    node_type_t_d nodes_d;
    node_type_t_h nodes_h;

    edge_type_t_d edges_d;
    edge_type_t_h edges_h;
    
    //current this defaults to default space
    count_t_d current_size;

    Graph(size_t n)
        : graph_d("Graph on the Device", n)
        , graph_h("Graph on the Host", n)
        , positions_d(Cabana::slice<0>(graph_d))
        , positions_h(Cabana::slice<0>(graph_h))
        , nodes_d(Cabana::slice<1>(graph_d))
        , nodes_h(Cabana::slice<1>(graph_h))
        , edges_d(Cabana::slice<2>(graph_d))
        , edges_h(Cabana::slice<2>(graph_h))
        , current_size("Graph Size")
    {
        current_size() = n;
    }

    void reslice () {
        positions_d = Cabana::slice<0>(graph_d);
        positions_h = Cabana::slice<0>(graph_h);

        nodes_d = Cabana::slice<1>(graph_d);
        nodes_h = Cabana::slice<1>(graph_h);

        edges_d = Cabana::slice<2>(graph_d);
        edges_h = Cabana::slice<2>(graph_h);
    }

    void copy_host_to_device() {
        Cabana::deep_copy(graph_d, graph_h);
        reslice();
    }

    void copy_device_to_host() {
        Cabana::deep_copy(graph_h, graph_d);
        reslice();
    }

    size_t capacity() {
        return graph_d.capacity();
    }
    
    //graph needs insertions 
    //
    //Pushes a point to the end of our graph atomically
    KOKKOS_INLINE_FUNCTION
    void atomic_push() const { //Not a bad choice if we have infrequent parallel pushes
        size_t id = Kokkos::atomic_fetch_add(&current_size(), 1); 
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
