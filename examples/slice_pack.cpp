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


template <class MemberTypes, class DeviceType>
struct MyWrapper {
	
	using my_type_t = Cabana::AoSoA<MemberTypes, DeviceType>;
    using sequence = decltype(std::make_integer_sequence<int, MemberTypes::size>{}); 
    
    template<int ... Ns>
    auto makeSlicePack(std::integer_sequence<int,Ns...>, my_type_t aosoa) {
        return Cabana::makeParameterPack( Cabana::slice<Ns>(aosoa)... );
    }

    //using pack_type = decltype(makeSlicePack(sequence{}, my_type_t{}));

	//my_type_t data;
    //pack_type slice_pack;

	/*MyWrapper(size_t num_tuple)
	  : data("MyData", num_tuple)
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
	}*/
    
     
};
//using my_type_t = Cabana::AoSoA<Cabana::MemberTypes<int, int>, DeviceType>; 
//template<int ... Ns>
//auto makeSlicePack(std::integer_sequence<int,Ns...>, my_type_t aosoa) {
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
    
    using MemberTypes = Cabana::MemberTypes<int, double[3]>;
     using sequence = decltype(std::make_integer_sequence<int, MemberTypes::size>); 
       
    //MyWrapper<MemberTypes, DeviceType> item; //item(10);

   /* my_type_t aosoa("aosoa", 10);
    auto my_pack = Cabana::makeParameterPack(Cabana::slice<0>(aosoa), Cabana::slice<1>(aosoa));
    auto slice0 = Cabana::get<0>(my_pack);
    for(auto i = 0; i < 10; i++) {
        slice0(i) = i;
    }
    using pack_type = decltype(makeSlicePack(std::make_integer_sequence<int, 2>{}, my_type_t{}));
    pack_type test_pack = my_pack;
    auto slice_test = Cabana::get<0>(my_pack);
    std::cout << "Slice Data: ";
    for(auto i = 0; i < 10; i++) {
        std::cout << slice_test(i) << " ";
    } std::cout << std::endl;*/
    //Cabana::makeParameterPack();    
    //MyWrapper<DeviceType> item(10);
    //item.slice_all();
    //using sequence = std::integer_sequence<int,2>;
    //auto my_pack = Cabana::makeParameterPack<int, int, double>(1, 2, 2.3);
    //std::cout << my_pack<0>::value_type << std::endl;
   // using pack_type = decltype(makeSlicePack(sequence{},my_type_t{}));
    return 0;
}
