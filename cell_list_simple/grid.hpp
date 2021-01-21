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
                  const Real delta_x, const Real delta_y)
            : _min_x(min_x)
            , _min_y(min_y)
            , _max_x(max_x)
            , _max_y(max_y)
        {
            //The grid construction is smart in the sense that it will
            //make the grid cells a bit larger so that we can have our
            //desired grid size
            _num_ee = cartesianCellsBetween(max_x, min_x, 1.0/delta_x);
            _num_eo = _num_ee+1;
            _num_r  = cartesianCellsBetween(max_x, min_x, 1.0/delta_y);
            _dx = (max_x - min_x)/_num_ee;
            _dy = (max_y - min_y)/_num_r;
            _rdx = 1.0/_dx;
            _rdy = 1.0/_dy;
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
      
        // Return the cardianal index of an ij cell index
        KOKKOS_INLINE_FUNCTION
        int cardinalCellIndex(const int i, const int j) const {
            if(j%2) { 
                return ((j-1)/2)*_num_ee+((j-1)/2)*_num_eo+_num_ee+i;
            } else { //EVEN
                return (j/2)*_num_ee+(j/2)*_num_eo+i;
            }
        }

        KOKKOS_INLINE_FUNCTION
        void locatePoint(const Real xp, const Real yp, int& ic, int& jc) const {
            //we first need to determine the row then we can locate the point
            jc = floor((yp - _min_y) / _dy);
            if(jc%2) { //ODD
                if(xp < _dx/2) {
                    ic = 0;
                } else if (xp > _max_x - _dx/2) {
                    ic = _num_eo - 1;
                } else {
                    ic = floor((xp - (_min_x + _dx/2))/_dx)+1;
                }
            } else { //EVEN
                ic = floor((xp - _min_x) / _dx);
            }
        }

        KOKKOS_INLINE_FUNCTION
        int cartesianCellsBetween(const Real max, const Real min, const Real rdelta) const {
            return floor((max-min)*rdelta);
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
