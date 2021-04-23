#include <Cabana_Core.hpp>
#include <iostream>
#include <string>

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


//The make slice pack functions are free functions, but they 
//could be written as static functions inside the class
//
//Implemenation function
template<int ... Ns, class AoSoA_t>
auto makeSlicePackImpl(std::integer_sequence<int,Ns...>, AoSoA_t aosoa) {
    return Cabana::makeParameterPack( Cabana::slice<Ns>(aosoa)... );
}

//Wrapper function
template<class AoSoA_t>
auto makeSlicePack(AoSoA_t aosoa) {
    return makeSlicePackImpl(std::make_integer_sequence<int, AoSoA_t::member_types::size>{}, aosoa);
}

template <class MemberTypes, class DeviceType>
struct MyWrapper {
	
	using my_type_t = Cabana::AoSoA<MemberTypes, DeviceType>;
    using sequence = decltype(std::make_integer_sequence<int, MemberTypes::size>{}); 
    
    //template<int ... Ns>
    //auto static makeSlicePack(std::integer_sequence<int,Ns...>, my_type_t aosoa) {
    //    return Cabana::makeParameterPack( Cabana::slice<Ns>(aosoa)... );
    //}

    using pack_type_t = decltype(makeSlicePack(my_type_t{}));

	my_type_t data;
    pack_type_t slice_pack;

	MyWrapper(std::string name, size_t num_tuple)
	  : data(name, num_tuple)
	{
        slice_all();
	}

	//slices inside the class allow us to do tagged functor super simple
	KOKKOS_INLINE_FUNCTION
	void operator()(const int i) const {
        //auto my_slice = Cabana::get<0>(slice_pack);
        //my_slice(i) = 1;
	}

	void slice_all() {
        //slice_pack = makeSlicePack(std::make_integer_sequence<int, MemberTypes::size>{}, my_type_t{});
	    slice_pack = makeSlicePack(data);
    }
};

int main(int argv, char* argc[]) {
    
    Kokkos::ScopeGuard scope_guard(argv, argc);
    
    using MemberTypes = Cabana::MemberTypes<int, double[3]>;
    using sequence = decltype(std::make_integer_sequence<int, MemberTypes::size>{}); 
       
    MyWrapper<MemberTypes, DeviceType> item("MyItem", 10);
    
    Cabana::get<0>(item.slice_pack)(0) = 1;
    
    return 0;
}
