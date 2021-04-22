#include <Cabana_Core.hpp>

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


template <class DeviceType>
struct MyWrapper {
	using DataTypes = Cabana::MemberTypes<int, double[3]>;

	using my_type_t = Cabana::AoSoA<DataTypes, DeviceType>;
	using slice_0_t = typename my_type_t::template member_slice_type<0>;
	using slice_1_t = typename my_type_t::template member_slice_type<1>;

	my_type_t data;

	slice_0_t slice0;
	slice_1_t slice1;

	MyWrapper(size_t num_tuple)
	  : data("MyData", num_tuple)
	{
		slice_all();
	}

	//slices inside the class allow us to do tagged functor super simple
	KOKKOS_INLINE_FUNCTION
	void operator()(const int i) const {
		slice0(i) = 1;
	}

	void slice_all() {
		slice0 = Cabana::slice<0>(data);
		slice1 = Cabana::slice<1>(data);
	}
    
    //template<int ... Ns>
    //auto makeSlicePack(std::integer_sequence<int,Ns...>, my_type_t) {
    //    return Cabana::makeParameterPack( Cabana::slice<Ns>(my_type_t)... );
    //}
};

//template<int ... Ns>
//auto makeSlicePack(std::integer_sequence<int,Ns...>, aosoa) {
//    return Cabana::makeParameterPack( Cabana::slice<Ns>(aosoa)... );
//}


//We can unpack parameters using fold expressions
//template<class... T> 
//auto add(T ...args)
//{
//    return (... + args); //means apply + operator to all arguments coming from pack args    
//}

template<class T, class...Args> 
auto make_unique(Args && ...args) 
{
    return std::unique_ptr<T>{new T(std::forward<Args>(args)...)};
}

template<class...Args>
struct Size {
    static const int value = sizeof...(Args);
};

template<typename T, T...ints>
void print_sequence(std::integer_sequence<T, ints...> int_seq) {
    std::cout << "The sequence of size " << int_seq.size() << ": ";
    //( (std::cout << ints << ' '), ... );
    std::cout << "\n";
}

//using sequence = decltype(std::make_integer_sequence<int,MemberTypes::size>);
int main(int argv, char* argc[]) {
    Kokkos::ScopeGuard scope_guard(argv, argc);
    //Cabana::makeParameterPack();    
    MyWrapper<DeviceType> item(10);
    item.slice_all();
    using sequence = std::integer_sequence<int,2>;

   // using pack_type = decltype(makeSlicePack(sequence{},my_type_t{}));
    return 0;
}
