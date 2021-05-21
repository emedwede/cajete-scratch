#ifndef CAJETE_NODETYPES_HPP
#define CAJETE_NODETYPES_HPP

#include <type_traits>
#include <string>

//Simple Verison of Graph Node Types

namespace Cajete {

//Type indexer
template<class T, int Size, int N, class Type, class... Types>
struct TypeIndexerImpl
{
    static constexpr std::size_t value =
        TypeIndexerImpl<T, Size, N-1, Types...>::value *
        (std::is_same<T, Type>::value ? Size - 1 - N : 1);
};

template <class T, int Size, class Type, class...Types>
struct TypeIndexerImpl<T, Size, 0, Type, Types...>
{
    static constexpr std::size_t value =
        std::is_same<T, Type>::value ? Size -1 : 1;
};

template<class T, class...Types>
struct TypeIndexer 
{
    static constexpr std::size_t index =
        TypeIndexerImpl<T, sizeof...(Types), sizeof...(Types)-1, Types...>::value;
};

namespace Field {
//Field Tags

//Forward declarations
template <class T>
struct Scalar;

//Scalar Field
struct ScalarBase {};

template<class T>
struct Scalar : ScalarBase {
    
    using value_type = T;
    static constexpr int rank = 0;
    static constexpr int size = 1;
    using data_type = value_type;

    template<class U>
    using field_type = Scalar<U>;
};

template <class T>
struct is_scalar_impl : std::is_base_of<ScalarBase, T> {};

template<class T>
struct is_scalar : is_scalar_impl<typename std::remove_cv<T>::type>::type {};

template<class View, class Layout>
struct ScalarViewWrapper {
    //TODO: add the scalar view wrapper
};


//Fields
struct VertexType : Scalar<int> {
    static std::string label() { return "vertex_type"; }
};

} //End namespace Field

} //End namespace Cajete

#endif
