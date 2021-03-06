
#ifndef NEW3_KNN_KERNEL_H
#define NEW3_KNN_KERNEL_H

namespace tensorflow {
namespace functor {

template<typename Device, typename dummy>
struct New3KnnOpFunctor {
    void operator()(
            const Device &d,

            const float *d_coord,
            const int* d_row_splits,
            int *d_indices,
            float *d_dist,

            const int n_vert,
            const int n_neigh,
            const int n_coords,

            const int n_rs,
            const bool tf_compat,
            const float max_radius,
            const size_t n_bins_x,
            const size_t n_bins_y
            );
};

}  // namespace functor
}  // namespace tensorflow

#endif //NEW3_KNN_KERNEL_H

