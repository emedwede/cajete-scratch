#ifndef __RESOURCE_HPP
#define __RESOURCE_HPP

#include <string>
#include <iostream>
#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>
namespace Cajete {

//Custom aosoa memory resource for an allocator
template<class DataTypes, class DeviceType>
class Resource {
    public:
        using view_type = Kokkos::View<size_t, DeviceType>;
        using size_type = size_t;
        using value_type = DataTypes;
        using device_type = DeviceType;
        using resource_type = Cabana::AoSoA<value_type, device_type>;
         
        Resource(std::string name, std::size_t s) 
            : data(name, s)
            , num_allocs("NumAllocs")
            , size("Size")
            , capacity("Capacity")
        {
            std::cout << "Constructing resource [ "
                << name << " , " << s << " ]\n";
        }
        
        KOKKOS_INLINE_FUNCTION
        size_type allocate(size_type numObjects) const {
            auto c = Kokkos::atomic_fetch_add(&size(), numObjects);
            Kokkos::atomic_increment(&num_allocs());
            return c;
        }

        resource_type get_data() {
            return data;
        }

        size_type get_num_allocs() {
            return num_allocs();
        }

        size_type get_capacity() {
            return data.size();
        }
    
        ~Resource() = default;

    private:
        view_type capacity;
        view_type  size;
        view_type num_allocs;
        resource_type data;

};

};

#endif
