
#ifndef NEW2_KNN_KERNEL_H
#define NEW2_KNN_KERNEL_H

namespace tensorflow {
namespace functor {

template<typename Device, typename dummy>
struct New2KnnOpFunctor {
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
            const int n_bins_x,
            const int n_bins_y
            );
};

}  // namespace functor
}  // namespace tensorflow

#endif //NEW2_KNN_KERNEL_H

