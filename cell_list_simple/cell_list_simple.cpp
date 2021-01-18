#include<iostream>

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>

//Linked Cell List

namespace Cajete {

template<class DeviceType>
class LLCellList {
    public:
        using device_type = DeviceType;
        using memory_space = typename device_type::memory_space;
        using execution_space = typename device_type::execution_space;
        using size_type = typename memory_space::size_type;
        using OffsetView = Kokkos::View<size_type*, device_type>;

};

}

int main() {
    using MemorySpace = Kokkos::DefaultExecutionSpace::memory_space;
    using ExecutionSpace = Kokkos::DefaultExecutionSpace;
    using DeviceType = Kokkos::Device<ExecutionSpace, MemorySpace>;

    using NodeTypes = Cabana::MemberTypes<double[3], size_t>;
    
    Cajete::LLCellList<DeviceType> cell_list;
    std::cout << "Hello simple\n";
}
