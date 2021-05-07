#ifndef __AOSOA_RESOURCE_HPP
#define __AOSOA_RESOURCE_HPP

#include <string>
#include <iostream>
#include <Cabana_Core.hpp>
#include "indexed_memory_resource.hpp"
#include "binner.hpp"

namespace Cajete {

//Custom aosoa memory resource for an allocator
template<class DataTypes, class DeviceType>
class AosoaResource : public IndexedMemoryResource {
    public:
        
        Binner<DeviceType> binner;

        using aosoa_resource_t = Cabana::AoSoA<DataTypes, DeviceType>;
 
        AosoaResource(std::string name, std::size_t s) 
            : data(name, s) 
        {
            std::cout << "Constructing resource [ "
                << name << " , " << s << " ]\n";
        }
        
        aosoa_resource_t get_resource() {
            return data;
        }

        size_t get_capacity() {
            return data.size();
        }
    
    
    private:
        IndexType do_allocate(IndexType n, IndexType p) override {
            printf("Allocation for aosoa resource\n");
            return 1;
        }

        void do_deallocate(IndexType n, IndexType p) override {
            printf("Deallocation for aosoa resource\n");
        }

        bool do_is_equal(const IndexedMemoryResource& rhs) const noexcept override {
            printf("Is equal for aosoa resource\n");
            return 0;
        }
        //Maybe the AoSoA should go here as private?
        aosoa_resource_t data;

};

};

#endif
