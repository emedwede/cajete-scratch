#ifndef __CAJETE_LL_CELL_LIST_HPP
#define __CAJETE_LL_CELL_LIST_HPP

#include<iostream>

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>

#include "grid.hpp"
#include "cell_complex.hpp"

namespace Cajete {

template<class DeviceType>
class LLCellList {
    public:
        using device_type = DeviceType;
        using memory_space = typename device_type::memory_space;
        using execution_space = typename device_type::execution_space;
        using size_type = typename memory_space::size_type;
        using OffsetView = Kokkos::View<size_type*, device_type>;

        LLCellList() {} //Default
        
        template<class SliceType>
        LLCellList(
            SliceType positions, const typename SliceType::value_type grid_delta[2],
            const typename SliceType::value_type grid_min[2],
            const typename SliceType::value_type grid_max[2],
            typename std::enable_if<( Cabana::is_slice<SliceType>::value ), int>::type* = 0)
            : _grid(grid_min[0], grid_min[1], 
                    grid_max[0], grid_max[1], 
                    grid_delta[0], grid_delta[1])
        {
            build(positions, 0, positions.size());
        }

        template<class SliceType>
        void build(SliceType positions, const std::size_t begin, 
                const std::size_t end) 
        {
            std::cout << "\n----------\nBuilding\n----------\n";
        }

    private:
        Cajete::BrickGrid2D<double> _grid;
};

}

#endif
