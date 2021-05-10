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
        
        struct TestTag {};
        using TestPolicy = Kokkos::RangePolicy<execution_space, TestTag>;


        BinningData() : num_bins(0) {}

        BinningData(int nbins, int bin_size)
            : num_bins(nbins)
            , counts("BinCounts", nbins)
            , offsets("BinOffsets", nbins)
            , capacities("BinCapacities", nbins)
        {
            Kokkos::deep_copy(capacities, bin_size);
           
            //const auto& _capacities = capacities;
            //const auto& _offsets = offsets;
            /*Kokkos::parallel_scan(nbins, KOKKOS_LAMBDA(const int i,
                        int &update, const bool final) {
                const int val = _capacities(i);
                if(final) {
                    _offsets(i) = update;
                }
                update += val;
            });*/

            //computeTest();
        }
        
        void computeTest() {
            TestPolicy policy_a(0, 10);
            Kokkos::parallel_for("Test", policy_a, *this);
        }

        KOKKOS_INLINE_FUNCTION
        void operator()(const TestTag&, const int i) const {
            printf("Here at bin %d\n", i);
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


template<class DeviceType>
 class Example {
public:
     struct TestTag {};
 
     using device = DeviceType;
     using memory_space = typename device::memory_space;
     using execution_space = typename device::execution_space;
     using HelloPolicy = Kokkos::RangePolicy<execution_space, TestTag>;
 
     Example() {
         computeTest();
     }
     
     void computeTest() {
         HelloPolicy policy_1(0, 10);
         Kokkos::parallel_for("Name", policy_1, *this);
    }   

    KOKKOS_INLINE_FUNCTION
     void operator()(const TestTag&, const int i) const {
         printf("Here at bin %d\n", i);
     }
};

};

#endif
