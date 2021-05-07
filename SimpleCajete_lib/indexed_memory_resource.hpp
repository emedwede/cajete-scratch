#ifndef INDEXED_MEMORY_RESOURCE_HPP
#define INDEXED_MEMORY_RESOURCE_HPP

/* A memory resource that works by using indirect
 * index access to blocks of memory instead of raw
 * pointers
 */

namespace Cajete {

class IndexedMemoryResource {
    //Interface to the Virtual Memory Resource Functions
    public:
        using IndexType = int;

        IndexType allocate(IndexType n, IndexType p) {
            return do_allocate(n, p);
        }

        void deallocate(IndexType n, IndexType p) {
            do_deallocate(n, p);
        }

        //needed to compare memory resources of the same type
        bool is_equal(const IndexedMemoryResource& rhs) const noexcept {
            return do_is_equal(rhs);
        }

        virtual ~IndexedMemoryResource() = default;

    //User must define the fuctions below in order to use the class
    private:
        virtual IndexType do_allocate(IndexType p, IndexType n) = 0;
        virtual void do_deallocate(IndexType p, IndexType n) = 0;
        virtual bool do_is_equal(const IndexedMemoryResource& rhs) const noexcept = 0;
};

};
#endif
