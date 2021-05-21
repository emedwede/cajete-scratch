#ifndef __CAJETE_NODELIST_HPP
#define __CAJATE_NODELIST_HPP

#include<Cabana_Core.hpp>

namespace Cajete {

//Node Traits:  here we use variadic templates to populate 
//              the tuple member types
template <class...FieldTags>
struct NodeTraits 
{
    using member_types = Cabana::MemberTypes<typename FieldTags::data_type...>;
};

template<class...FieldTags>
struct Node
{
    using traits = NodeTraits<FieldTags...>;
    using tuple_type = Cabana::Tuple<typename traits::member_types>;

    static constexpr int vector_length = 1;

    Node() = default;

    //Wrapper constructor for tuple
    KOKKOS_FORCEINLINE_FUNCTION
    Node(const tuple_type& tuple) : _tuple(tuple) {}

    //get the tuple
    KOKKOS_FORCEINLINE_FUNCTION
    tuple_type& tuple() {return _tuple;}

    KOKKOS_FORCEINLINE_FUNCTION
    const tuple_type& tuple() const { return _tuple; }

    //tuple wrapped
    tuple_type _tuple;
};

//view of nodes. wraps a view of the SOA a node resides in
template<int VectorLength, class... FieldTags>
struct NodeView
{
    using traits = NodeTraits<FieldTags...>;

    using soa_type = Cabana::SoA<typename traits::member_types, VectorLength>;

    static constexpr int vector_length = VectorLength;

    NodeView() = default;

    //wrapper constructor for soa around a tuple
    KOKKOS_FORCEINLINE_FUNCTION
    NodeView(soa_type& soa, const int vector_index) 
        : _soa(soa)
        , _vector_index(vector_index)
    {

    }

    //Get the SoA
    KOKKOS_FORCEINLINE_FUNCTION
    soa_type& soa() { return _soa; }

    const soa_type& soa() const { return _soa; }

    //get the vector index of the node in the SoA
    KOKKOS_FORCEINLINE_FUNCTION
    int vectorIndex() const { return _vector_index; }

    //soa the node is in
    soa_type& _soa;

    //local vector index of the node
    int _vector_index;
};
//accesor functions ----------------------------------------

//node accessor
template <class FieldTag, class... FieldTags, class... IndexTypes>
KOKKOS_FORCEINLINE_FUNCTION typename std::enable_if<
    sizeof...( IndexTypes ) == FieldTag::rank,
    typename Node<FieldTags...>::tuple_type::
        template member_const_reference_type<
            TypeIndexer<FieldTag, FieldTags...>::index>>::type
get( const Node<FieldTags...>& node, FieldTag, IndexTypes... indices )
{
    return Cabana::get<TypeIndexer<FieldTag, FieldTags...>::index>(
        node.tuple(), indices... );
}

template <class FieldTag, class... FieldTags, class... IndexTypes>
KOKKOS_FORCEINLINE_FUNCTION typename std::enable_if<
    sizeof...( IndexTypes ) == FieldTag::rank,
    typename Node<FieldTags...>::tuple_type::template member_reference_type<
        TypeIndexer<FieldTag, FieldTags...>::index>>::type
get( Node<FieldTags...>& node, FieldTag, IndexTypes... indices )
{
    return Cabana::get<TypeIndexer<FieldTag, FieldTags...>::index>(
        node.tuple(), indices... );
}

// NodeView accessor
template <class FieldTag, class... FieldTags, class... IndexTypes,
          int VectorLength>
KOKKOS_FORCEINLINE_FUNCTION typename std::enable_if<
    sizeof...( IndexTypes ) == FieldTag::rank,
    typename NodeView<VectorLength, FieldTags...>::soa_type::
        template member_const_reference_type<
            TypeIndexer<FieldTag, FieldTags...>::index>>::type
get( const NodeView<VectorLength, FieldTags...>& node, FieldTag,
     IndexTypes... indices )
{
    return Cabana::get<TypeIndexer<FieldTag, FieldTags...>::index>(
        node.soa(), node.vectorIndex(), indices... );
}

template <class FieldTag, class... FieldTags, class... IndexTypes,
          int VectorLength>
KOKKOS_FORCEINLINE_FUNCTION typename std::enable_if<
    sizeof...( IndexTypes ) == FieldTag::rank,
    typename NodeView<VectorLength, FieldTags...>::soa_type::
        template member_reference_type<
            TypeIndexer<FieldTag, FieldTags...>::index>>::type
get( NodeView<VectorLength, FieldTags...>& node, FieldTag,
     IndexTypes... indices )
{
    return Cabana::get<TypeIndexer<FieldTag, FieldTags...>::index>(
        node.soa(), node.vectorIndex(), indices... );
}
//----------------------------------------------------------

template <class DeviceType, class... FieldTags>
class NodeList 
{
    public:
        
        using memory_space = typename DeviceType::memory_space;

        using traits = NodeTraits<FieldTags...>;

        using aosoa_type = Cabana::AoSoA<typename traits::member_types, memory_space>;

        using tuple_type = typename aosoa_type::tuple_type;

        //slicer
        template<std::size_t M>
        using slice_type = typename aosoa_type::template member_slice_type<M>;

        using node_type = Node<FieldTags...>;
        
        using node_view_type = NodeView<aosoa_type::vector_length, FieldTags...>;

        //defualt constructor
        NodeList(const std::string& label) 
            : _aosoa(label)
        {
            //TODO: set partition information
        }

        //get the capacity of the node list
        std::size_t capacity() const { return _aosoa.size(); }

        //return the underlying AoSoA of nodes
        aosoa_type& aosoa() { return _aosoa; }

        const aosoa_type& aosoa() const { return _aosoa; }

        KOKKOS_INLINE_FUNCTION
        node_type get_node(const std::size_t p) const 
        {
            return _aosoa.getTuple(p);
        }

        KOKKOS_INLINE_FUNCTION
        void insert(const std::size_t p, node_type& node) const 
        {
            _aosoa.setTuple(p, node.tuple());
        }
        //get the slice of a given field
        template <class FieldTag>
        slice_type<TypeIndexer<FieldTag, FieldTags...>::index> slice(FieldTag) const {
            return Cabana::slice<TypeIndexer<FieldTag, FieldTags...>::index>(_aosoa, FieldTag::label());
        }
        
    private:
        aosoa_type _aosoa;
};

} //end namespace Cajete

#endif
