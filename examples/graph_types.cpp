#include <Cabana_Core.hpp>

#include <iostream>
#include <string>

//The general type indexer
template <class T, int Size, int N, class Type, class... Types>
struct TypeIndexerImpl {
    static constexpr std::size_t value =
        TypeIndexerImpl<T, Size, N - 1, Types...>::value *
        ( std::is_same<T, Type>::value ? Size - 1 - N : 1 );
};

template <class T, int Size, class Type, class... Types>
struct TypeIndexerImpl<T, Size, 0, Type, Types...> {
    static constexpr std::size_t value =
        std::is_same<T, Type>::value ? Size - 1 : 1;
};

template <class T, class... Types>
struct TypeIndexer {
    static constexpr std::size_t index = 
        TypeIndexerImpl<T, sizeof...( Types ), sizeof...( Types ) - 1, Types...>::value;
};
namespace Field {

    //foward delcaration

    template<class T>
    struct SimpleScalar;

    //struct SimpleScalarBase {};
    
    template <class T>
    struct SimpleScalar {
        using value_type = T;

        using data_type = value_type;
    };

};

struct TestType : Field::SimpleScalar<int> {
    static std::string label() { return "simple_test_type"; }
};

template <class... FieldTags>
struct NodeTraits {
    using member_types = Cabana::MemberTypes<typename FieldTags::data_type...>;
};

template <class... FieldTags>
struct Node {
    
    using traits = NodeTraits<FieldTags...>;

    //get the tuple traits from the template parameter pack
    using tuple_type = Cabana::Tuple<typename traits::member_types>;
   
    static constexpr int vector_length = 1;

    Node() = default;

    //Tuple wrapper constructor
    KOKKOS_FORCEINLINE_FUNCTION
    Node(const tuple_type& tuple) : _tuple(tuple) {}

    //Get the underlying tuple
    KOKKOS_FORCEINLINE_FUNCTION
    tuple_type& tuple() { return _tuple; };

    //TODO: What does this version do?
    KOKKOS_FORCEINLINE_FUNCTION
    const tuple_type& tuple() const { return _tuple; }

    //The tuple type wrapped by this node
    tuple_type _tuple;
};

template <int VectorLength, class... FieldTags>
struct NodeView {
    using traits = NodeTraits<FieldTags...>;
    using soa_type = Cabana::SoA<typename traits::member_types, VectorLength>;

    //Default NodeView constructor
    NodeView() = default;

    //Tuple wrapper constructor for view
    KOKKOS_FORCEINLINE_FUNCTION
    NodeView(soa_type& soa, const int vector_index) 
        : _soa(soa)
        , _vector_index(vector_index)
    {

    }
 
    //get the underlying SoA
    KOKKOS_FORCEINLINE_FUNCTION
    soa_type& soa() {return _soa;}

    KOKKOS_FORCEINLINE_FUNCTION
    const soa_type& soa() const {return _soa;}

    //get the vector index
    int vectorIndex() const { return _vector_index; };

    //The soa of the the node
    soa_type& _soa;

    //The local index of the node
    int _vector_index;
};

template <class CellComplex, class... FieldTags>
class NodeList {

public:

    //possibly do the cell complex thing here
    using cell_complex_type = CellComplex;
    
    using memory_space = typename CellComplex::memory_space;

    //alias away the node traits
    using traits = NodeTraits<FieldTags...>;
    
    using aosoa_type = Cabana::AoSoA<typename traits::member_types, memory_space>;

    using tupe_type = typename aosoa_type::tuple_type;

    template <std::size_t M>
    using slice_type = typename aosoa_type::template member_slice_type<M>;

    using node_type = Node<FieldTags...>;

    using node_view_type = NodeView<aosoa_type::vector_length, FieldTags...>;
    
    //Default constructor, starts the NodeList at size 0?
    NodeList(const std::string& label) : _aosoa(label) {}

    //Get the number of of available nodes in the list
    std::size_t size() const {return _aosoa.size(); }

    //Get the capacity
    std::size_t capacity() const {return _aosoa.capacity(); }

    //get the aosoa
    aosoa_type& aosoa() { return _aosoa; }

    const aosoa_type& aosoa() const {return _aosoa; }

    //get a slice of a give field!
    template <class FieldTag>
    slice_type<TypeIndexer<FieldTag, FieldTags...>::index>
        slice(FieldTag) const {
        return Cabana::slice<TypeIndexer<FieldTag, FieldTags...>::index>(_aosoa, FieldTag::label() );
    }
    
    template <std::size_t M>
    slice_type<M> slice() const {
        return Cabana::slice<M>(_aosoa, "MyLabel");
    }
private:
    aosoa_type _aosoa;

};

template<typename DeviceType>
struct CellComplex {
    using memory_space = typename DeviceType::memory_space;
};

int main (int argc, char * argv[]) {
    Kokkos::ScopeGuard scope_guard(argc, argv);

    std::cout << "Cabana Types Test" << std::endl;
    
    CellComplex<Kokkos::DefaultExecutionSpace> cplex;

    //Goal: we want to get to this
    using list_type = NodeList<CellComplex<Kokkos::DefaultExecutionSpace>, 
          Field::SimpleScalar<int>, TestType>;
    list_type nodes("test_nodes");
        
    std::cout << "Aosoa Size, Capacity: " << nodes.size() << " " << nodes.capacity() << std::endl;
   
    auto& aosoa = nodes.aosoa();
    aosoa.resize(33);
    std::cout << "Resized Aosoa Size, Capacity: " << nodes.size() << " " << nodes.capacity() << std::endl;
   

    auto ss_0 = nodes.slice<0>();
    std::cout << "Slice size: " << ss_0.size() << std::endl;

    auto ss_1 = nodes.slice( TestType() );
    std::cout << "Slice size: " << ss_1.size() << std::endl;

    Kokkos::parallel_for("Test", 10, KOKKOS_LAMBDA(const int p) {
        typename list_type::node_type node( aosoa.getTuple(p) ); 
    });

    Cabana::Tuple<Cabana::MemberTypes<double[3], int>> tuple;

    Cabana::get<1>(tuple) = 3;
    
    int a = Cabana::get<1>(tuple);
    
    std::cout << "Tup: " << a << std::endl;
    
    std::cout << "Num of Node SOA: " << ss_1.numSoA() << std::endl;
    return 0;
}
