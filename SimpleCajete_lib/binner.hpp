#ifndef __BINNER_HPP
#define __BINNER_HPP

#include <Kokkos_Core.hpp>

namespace Cajete {

/* Holds all the basic information to partition and bin
 * data in a 1D array space
 */

template <class DeviceType>
class BinningData {
    public:
        using device_type = DeviceType;
        using memory_space = typename device_type::memory_space;
        using execution_space = typename device_type::execution_space;
        using size_type = typename memory_space::size_type;
        using CountView = Kokkos::View<size_type*, device_type>;
        using OffsetView = Kokkos::View<size_type*, device_type>;

        BinningData() : num_bins(0) {}

        BinningData(int nbins, int bin_size)
            : num_bins(nbins)
            , counts("BinCounts", nbins)
            , offsets("BinOffsets", nbins)
            , capacities("BinCapacities", nbins)
        {
            Kokkos::deep_copy(capacities, bin_size);
            
            Kokkos::parallel_scan(nbins, KOKKOS_LAMBDA(const int i,
                        int &update, const bool final) {
                const int val = capacities(i);
                if(final) {
                    offsets(i) = update;
                }
                update += val;
            });
        }

        BinningData(CountView _counts, OffsetView _offsets, OffsetView _capacities) 
            : counts(_counts)
            , offsets(_offsets)
            , capacities(_capacities)
            , num_bins(counts.extent(0))
        {
        
        }

        //get the number of bins
        KOKKOS_INLINE_FUNCTION
        int numBin() const { return num_bins; }

        //get the number of items in a bin
        KOKKOS_INLINE_FUNCTION
        int binSize( const size_type bin_id ) const { return counts( bin_id ); }

        //starting offset of a bin
        KOKKOS_INLINE_FUNCTION
        size_type binOffset( const size_type bin_id ) const
        {
            return offsets( bin_id );
        }

        //get the capacity of a bin
        KOKKOS_INLINE_FUNCTION
        size_type binCapacity(const size_type bin_id) const
        {
            return capacities( bin_id );
        }

    private:
        int num_bins;
        CountView counts;
        OffsetView offsets;
        OffsetView capacities;
};

};

#endif
