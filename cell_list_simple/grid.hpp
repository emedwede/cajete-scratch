#ifndef __CAJETE_BRICKGRID_HPP
#define __CAJETE_BRICKGRID_HPP

//#include "types.hpp"


namespace Cajete {

//Currently we have a 2D version, but later I'd like to update it to 3D
template <class Real, typename std::enable_if<std::is_floating_point<Real>::value, int>::type = 0>
class BrickGrid2D {
    public: 
        using Dims = size_t;
        using real_type = Real;
        Real _min_x;
        Real _min_y;
        Real _max_x;
        Real _max_y;
        Real _dx;
        Real _dy;
        Real _rdx;
        Real _rdy;
        Dims _num_r; //number of rows
        //Even starts at "zero"
        Dims _num_ee; //number of elements in an even row
        Dims _num_eo; //number of elements in an odd row
        
        BrickGrid2D() {}
        
        //We don't have to set the number of elements in
        //even and odd rows as the number of rows, but for
        //convenience we do
        BrickGrid2D(const Real min_x, const Real min_y, 
                  const Real max_x, const Real max_y,
                  const Real dx, const Real dy,
                  const Dims num_r)
            : _min_x(min_x)
            , _min_y(min_y)
            , _max_x(max_x)
            , _max_y(max_y)
            , _dx(dx)
            , _dy(dy)
            , _num_r(num_r)
            , _num_ee(num_r)
            , _num_eo(num_r+1)
        {
            //do more stuff here
            //later we need to add a check that num_r > 1
        }
        KOKKOS_INLINE_FUNCTION
        std::size_t totalNumCells() const {
            //something like this, in cartesian it's nx*ny*nz
            //for brick grid it's number of n rows + number n+1 rows
            if(_num_r%2) { //ODD
                return  ((_num_r-1)/2)*_num_ee+((_num_r-1)/2)*_num_eo+_num_ee; 
            } else{ //EVEN
                return (_num_r/2)*_num_ee+(_num_r/2)*_num_eo; 
            }
        }
       
        //TODO: Finish and correct locate point
        KOKKOS_INLINE_FUNCTION
        void locatePoint(const Real xp, const Real yp, int& ic, int& jc) const {
            //we first need to determine the row then we can locate the point
            jc = floor((yp - _min_y) * 1/_dy);
            if(jc%2) { //ODD
                ic = floor((xp - _min_x) / _dx);
            } else { //EVEN
                if(xp < _dx/2) {
                    ic = 0;
                } else if (xp > _dx/2) {
                    ic = _num_ee - 1;
                }
            }
        }

        //size_t compute_num_2D_zones(); <==> same as totalNumCells() in 2D
        //size_t compute_num_1D_zones();
        //size_t compute_num_0D_zones();

};

//rectangular case based off the alternating, n/n+1 row criteria for 
//laying bricks
/*size_t BrickGrid::compute_num_1D_zones() {
    if(n_rows%2) {
        return ((n_rows-1)/2)*(n_elem-1)+((n_rows-1)/2)*n_elem+(n_elem-1)+(n_rows-1)*(2*n_elem); 
    } else {
        return (n_rows/2)*(n_elem-1)+(n_rows/2)*n_elem+(n_rows-1)*(2*n_elem);
    }
}*/

//rectangular case based off the alternating, n/n+1 row criteria for 
//laying bricks
/*size_t BrickGrid::compute_num_0D_zones() {
    return (2*n_elem-1)*(n_rows-1);
}*/

} // end namespace Cajete

#endif
