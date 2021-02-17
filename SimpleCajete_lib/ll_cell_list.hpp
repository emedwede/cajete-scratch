#ifndef __CAJETE_LL_CELL_LIST_HPP
#define __CAJETE_LL_CELL_LIST_HPP

#include<iostream>

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>

#include "grid.hpp"
#include "cell_complex.hpp"
#include "graph.hpp"

namespace Cajete {

template<class DeviceType>
class StrataData {

public:
    using device = DeviceType;
    using memory_space = typename device::memory_space;
    using execution_space = typename device::execution_space;
    //using size_type = typename memory_space::size_type;

    using view1D_d_t = Kokkos::View<long int*, memory_space>; 
    using view1D_h_t = typename view1D_d_t::HostMirror;
    
    StrataData() : _nbin_g(0), _nbin_l(0) {std::cout << "Default construction of StrataData\n";}

    StrataData( const std::size_t begin, 
                const std::size_t end,
                view1D_d_t counts_g,
                view1D_d_t offsets_g,
                view1D_d_t permute_g,
                view1D_d_t counts_l,
                view1D_d_t offsets_l,
                view1D_d_t permute_l)
        : _begin(begin)
        , _end(end)
        , _nbin_g(counts_g.extent(0))
        , _nbin_l(counts_l.extent(0))
        , _counts_d(counts_g)
        , _offsets_d(offsets_g)
        , _permute_d(permute_g)
        , _counts_l_d(counts_l)
        , _offsets_l_d(offsets_l)
        , _permute_l_d(permute_l)
    {}


    KOKKOS_INLINE_FUNCTION
    int numBinGlobal() const { return _nbin_g; }

    KOKKOS_INLINE_FUNCTION
    int numBinLocal() const { return _nbin_l; }

    KOKKOS_INLINE_FUNCTION
    int binCountsGlobal(const long int bin_id) const {
        return _counts_d(bin_id);
    }

    KOKKOS_INLINE_FUNCTION
    int binCountsLocal(const long int bin_id) const {
        return _counts_l_d(bin_id);
    }

    KOKKOS_INLINE_FUNCTION
    long int binOffsetGlobal(const long int bin_id) const {
        return _offsets_d(bin_id);
    }

    KOKKOS_INLINE_FUNCTION
    long int binOffsetLocal(const long int bin_id) const {
        return _offsets_l_d(bin_id);
    }

    //TODO: add a function to compute bin size
    KOKKOS_INLINE_FUNCTION
    int binSizeGlobal(const long int bin_id) const {
        //TODO
    }

    KOKKOS_INLINE_FUNCTION
    int binSizeLocal(const long int bin_id) const {
        //TODO
    }

    KOKKOS_INLINE_FUNCTION
    long int permutationGlobal(const long int tuple_id) const {
        return _permute_d(tuple_id);
    }

    KOKKOS_INLINE_FUNCTION
    long int permutationLocal(const long int tuple_id) const {
        return _permute_l_d(tuple_id);
    }


private:
    std::size_t _begin;
    std::size_t _end;
    int _nbin_g, _nbin_l;


    view1D_d_t _counts_d;
    view1D_h_t _counts_h;
    view1D_d_t _offsets_d;
    view1D_h_t _offsets_h;
    view1D_d_t _permute_d;
    view1D_h_t _permute_h;

    view1D_d_t _counts_l_d;
    view1D_h_t _counts_l_h;
    view1D_d_t _offsets_l_d;
    view1D_h_t _offsets_l_h;
    view1D_d_t _permute_l_d;
    view1D_h_t _permute_l_h;
};

template<class DeviceType>
class LLCellList {
    public:
        using device_type = DeviceType;
        using memory_space = typename device_type::memory_space;
        using execution_space = typename device_type::execution_space;
        using size_type = typename memory_space::size_type;
        using OffsetView = Kokkos::View<size_type*, device_type>;

        StrataData<DeviceType> layered_bin_data;
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
