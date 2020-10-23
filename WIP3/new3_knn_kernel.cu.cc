//#define GOOGLE_CUDA 1
//
#define BLOCK_SIZE 256

#if GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "new3_knn_kernel.h"
#include "helpers.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "cuda_helpers.h"
#include <iostream>

namespace tensorflow {
namespace functor {

namespace cpu{

void set_defaults(
        int *d_indices,
        float *d_dist,
        const bool tf_compat,
        const int n_vert,
        const int n_neigh
){
    for(size_t i = 0 ; i < n_vert*n_neigh ; i += 1){
        d_indices[i] = -1;
        d_dist[i] = 0;
    }
}


void print_array(
        const float *in_arr,
        const size_t start,
        const size_t end
){
    for(size_t i = start ; i < end ; i += 1){
        float tmp_val = in_arr[i];
        printf("i: %d;\t%f\n", i, tmp_val);
    }
}
void print_array(
        const int *in_arr,
        const size_t start,
        const size_t end
){
    for(size_t i = start ; i < end ; i += 1){
        float tmp_val = in_arr[i];
        printf("i: %d;\t%f\n", i, tmp_val);
    }
}

}// cpu namespace

namespace gpu{
__device__
float calculateDistance(size_t i_v, size_t j_v, const float * d_coord, size_t n_coords){
    float distsq=0;
    if(i_v == j_v)
        return 0;
    for(size_t i=0;i<n_coords;i++){
        float dist = d_coord[I2D(i_v,i,n_coords)] - d_coord[I2D(j_v,i,n_coords)];
        distsq += dist*dist;
    }
    return distsq;
}


__device__
int searchLargestDistance(int i_v, float* d_dist, int n_neigh, float& maxdist){
    maxdist=0;
    int maxidx=0;
    for(size_t n=0;n<n_neigh;n++){
        float distsq = d_dist[I2D(i_v,n,n_neigh)];
        if(distsq > maxdist){
            maxdist = distsq;
            maxidx = n;
        }
    }
    return maxidx;
}

__device__
float calculate2dDistanceToThePoint(float *pointCoord, size_t i_v, const float* d_coord, size_t n_coords){
    float distsq=0;
    for(size_t i=0;i<2;i++){
        float dist = d_coord[I2D(i_v,i,n_coords)] - pointCoord[i];
        distsq += dist*dist;
    }
    return distsq;
}

__device__
int getGlobalIdx_2D_2D(){
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int threadId = blockId * (blockDim.x * blockDim.y)
         + (threadIdx.y * blockDim.x) + threadIdx.x;
    return threadId;
}


__global__
void set_defaults(
        int *d_indices,
        float *d_dist,
        const bool tf_compat,
        const int n_vert,
        const int n_neigh
){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(size_t i = index ; i < n_vert*n_neigh ; i += stride){
        d_indices[i] = -1;
        d_dist[i] = 0;
    }
}

template <typename T>
__global__
void set_defaults(
        T *in_arr,
        const size_t arr_size,
        const T def_val
){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(size_t i = index ; i < arr_size ; i += stride){
        in_arr[i] = def_val;
    }
}

template <typename T>
__device__
void set_defaults_device(
        T *in_arr,
        const size_t arr_size,
        const T def_val
){
    int index = 0;
    int stride = 1;
    for(size_t i = index ; i < arr_size ; i += stride){
        in_arr[i] = def_val;
    }
}

__global__
void print_neighbours(
        const size_t i_v,
        int *d_indices,
        float *d_dist,
        const size_t n_neigh
){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(size_t i = index ; i < n_neigh ; i += stride){
        printf(" %d (%f)", d_indices[I2D(i_v,i,n_neigh)], d_dist[I2D(i_v,i,n_neigh)]);
    }
}

template <typename T>__global__
void print_array(
        const T *in_arr,
        const size_t start,
        const size_t end
){
    int index = blockIdx.x * blockDim.x + threadIdx.x + start;
    int stride = blockDim.x * gridDim.x;
    for(size_t i = index ; i < end ; i += stride){
        printf("i: %d;\t%f\n", i, in_arr[i]);
    }
}


__global__
void get_bin_coords(float* min, float* max, float *coords_2d_bins, const size_t n_bins_x, const size_t n_bins_y){

    const size_t iBinX = threadIdx.x/n_bins_x;
    const size_t iBinY = threadIdx.x - iBinX*n_bins_x;

    // const size_t iBinX =  blockIdx.x * blockDim.x + threadIdx.x;
    if(iBinX >= n_bins_x)
        return;
    // const size_t iBinY =  blockIdx.y * blockDim.y + threadIdx.y;
    if(iBinY >= n_bins_y)
        return;

    printf("get_bin_coords:\tn_bins_x: %d;\tn_bins_y:%d\n", n_bins_x, n_bins_y);

    // define phase-space bin edges
    size_t bin_index = I2D(iBinX, iBinY, n_bins_y);
    coords_2d_bins[4*bin_index] = min[0] + iBinX*(max[0] - min[0])/n_bins_x;
    coords_2d_bins[4*bin_index+1] = min[1] + iBinY*(max[1] - min[1])/n_bins_y;
    coords_2d_bins[4*bin_index+2] = min[0] + (iBinX+1)*(max[0] - min[0])/n_bins_x;
    coords_2d_bins[4*bin_index+3] = min[1] + (iBinY+1)*(max[1] - min[1])/n_bins_y;
}

// calculate min and max of X and Y coordinates among all vertices
// make bins with widths: (x_max-x_min)/n_bins and (y_max-y_min)/n_bins
__global__
void constructPhaseSpaceBins(const float *d_coord, size_t n_coords, size_t start_vert,
                             size_t end_vert, size_t* n_bins,
                             float* min, float* max, int *n_vertices_in_bin,
                             size_t *indices_of_vertices_in_bin){

    // define which vertices belong to which bin
    int index = blockIdx.x * blockDim.x + threadIdx.x + start_vert;
    int stride = blockDim.x * gridDim.x;
    for(size_t i_v = index; i_v < end_vert; i_v += stride){
        
        size_t iDim = 0;
        float coord = d_coord[I2D(i_v,iDim,n_coords)];
        size_t indx_1 = (size_t)((coord - min[iDim])/((max[iDim]-min[iDim])/n_bins[iDim]));
        iDim = 1;
        coord = d_coord[I2D(i_v,iDim,n_coords)];
        size_t indx_2 = (size_t)((coord - min[iDim])/((max[iDim]-min[iDim])/n_bins[iDim]));

        size_t bin_index = I2D(indx_1, indx_2, n_bins[1]);
        atomicAdd(&n_vertices_in_bin[bin_index], 1); // to avoid race condition
        indices_of_vertices_in_bin[i_v] = bin_index;
    }

}

__device__
void prepare_indices(const size_t start_vert, const size_t end_vert, const size_t bin_index, const size_t* indices_of_vertices_in_bin, int* counters, int* tmp_indices){
    int index = start_vert;
    int stride = 1;
    for(size_t i_v = index; i_v < end_vert; i_v += stride){
        if (indices_of_vertices_in_bin[i_v] != bin_index)
            return;
        int index_to_fill = atomicAdd(&counters[0],1);
        tmp_indices[index_to_fill] = i_v;
    }
}

__global__
void clean_indices(size_t bin_index, int* tmp_indices, int *n_vertices_in_bin, int defValue){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(size_t i_v = index; i_v < n_vertices_in_bin[bin_index]; i_v += stride){
        tmp_indices[i_v] = defValue;
    }
}

__device__
void findNeighbours(const int* indices_of_vert_to_find_new_neigh, // vertices for which we want to find neighbours in the targe phase-space bin
                    const size_t n_vertices_to_loop, // size of the first input array
                    const size_t indx_bin_to_use, // index of the newly added bin
                    const size_t* index_map_to_bins, // index_map_to_bins[i_v] -> bin number to which vertex belong
                    const int* n_vertices_in_bin,
                    const float *d_coord,
                    size_t start_vert,
                    size_t end_vert,
                    size_t n_coords, // number of dimentions
                    size_t n_neigh, // number of neighbours
                    float* d_dist, // distance matrix
                    int* d_indices, // indices matrix which corresponds to distance one
                    float max_radius = -1.0 // max. radius to search for neighbours
                    ){
    
    // loop to assign indices and distances to other vertices
    int index = 0;
    int stride = 1;
    for(size_t i = index; i <  n_vertices_to_loop; i += stride){
    // for(size_t i = 0; i < n_vertices_to_loop; i++){
        size_t i_v = indices_of_vert_to_find_new_neigh[i];

        //protection against n_vert<n_neigh
        size_t max_neighbours = n_neigh;

        size_t nfilled=0;
        int running_index = max_neighbours - 1;

        while (running_index>=0){
            if (d_indices[I2D(i_v,running_index,n_neigh)] == -1) // default init value
                running_index -= 1;
            else{
                nfilled = running_index+1;
                break;
            }
        }
        
        //set default to self
        if((n_vertices_in_bin[indx_bin_to_use]+nfilled)<n_neigh){
            max_neighbours=(n_vertices_in_bin[indx_bin_to_use]+nfilled);
        }
        
        float maxdistsq = 0;
        size_t maxidx_local = 0;
        if (nfilled>0){
            maxidx_local = searchLargestDistance(i_v,d_dist,n_neigh,maxdistsq);
        }
        
        
        // assigning loop - searching neighbouth for i_v
        for(size_t j_v=start_vert;j_v<end_vert;j_v++){
            if(index_map_to_bins[j_v]!=indx_bin_to_use)
                continue;
            //fill up
            float distsq = calculateDistance(i_v,j_v,d_coord,n_coords);
          
            if(nfilled<max_neighbours && (max_radius<=0 || max_radius>=distsq)){
                // filling in distances until we reach max_neighbours
                d_indices[I2D(i_v,nfilled,n_neigh)] = j_v;
                d_dist[I2D(i_v,nfilled,n_neigh)] = distsq;
                
                if(distsq > maxdistsq){
                    maxdistsq = distsq;
                    maxidx_local = nfilled;
                }
                nfilled++;
                continue;
            }
            
            // if we already filled max_neighbours distances, compare each new distance
            // with the current maximum. if distance is smaller - threw away current maximum,
            // fill in new distance and find new maximum
            if(distsq < maxdistsq){// automatically applies to max radius
                //replace former max
                d_indices[I2D(i_v,maxidx_local,n_neigh)] = j_v;
                d_dist[I2D(i_v,maxidx_local,n_neigh)] = distsq;

                //search new max
                maxidx_local = searchLargestDistance(i_v,d_dist,n_neigh,maxdistsq);
            }
        }// loop through vertices
    }// loop through vertices
}

__device__
void calculate2dDistanceToTheBinEdges(
    int* output_indices, // ???
    int* n_output_vertices, // ???
    const int* input_indices, // vertices for which we want to find neighbours in the targe phase-space bin
    const size_t n_input_vertices, // size of the first input array
    const float* target_bin_coords, // {x1, y1, x2, y2} coords of the target bin
    const float *d_coord,
    const size_t n_coords, // number of dimentions
    size_t n_neigh, // number of neighbours
    float* d_dist, // distance matrix
    int* d_indices, // indices matrix which corresponds to distance one
    float max_radius // max. radius to search for neighbours
    ){


    int index = 0;
    int stride = 1;
    for(size_t i = index; i < n_input_vertices; i += stride){
        size_t i_v = input_indices[i];
        
        // safety check
        float x = d_coord[I2D(i_v,0,n_coords)];
        float y = d_coord[I2D(i_v,1,n_coords)];
        if (x>target_bin_coords[0] && x<target_bin_coords[2]
            && y>target_bin_coords[1] && y<target_bin_coords[3]){
            continue; // i_v belongs to the target bin 
        }
        
        // check if i_v has required number of neighbours:
        size_t n_found_neighbours=0;
        int running_index = n_neigh - 1;
        while (running_index>=0){
            if (d_indices[I2D(i_v,running_index,n_neigh)] == -1) // default init value
                running_index -= 1;
            else{
                n_found_neighbours = running_index+1;
                break;
            }
        }
                
        // include i_v for output if it doesn't have enough neighbours
        if (n_found_neighbours<n_neigh){
            output_indices[atomicAdd(&n_output_vertices[0],1)] = i_v;
            continue;
        }
        
        // find the distance to the farthermost neighbour
        float maxdistsq = 0; // largest distance
        size_t maxidx_local = searchLargestDistance(i_v,d_dist,n_neigh,maxdistsq);
        
        // find the closest distance to the target bin
        float distance_to_bin = 0.0;
        if ((x>target_bin_coords[0] && x<target_bin_coords[2]) || 
            (y>target_bin_coords[1] && y<target_bin_coords[3])){
            size_t iDim = 0;
            if (x>target_bin_coords[0] && x<target_bin_coords[2])
                iDim = 1;
            
            float lowBinEdge = target_bin_coords[iDim];
            float highBinEdge = target_bin_coords[iDim+2];
            float d1 = pow((d_coord[I2D(i_v,iDim,n_coords)] - lowBinEdge),2);
            float d2 = pow((d_coord[I2D(i_v,iDim,n_coords)] - highBinEdge),2);
            distance_to_bin = (d1<d2) ? d1 : d2;
        }
        else{ // diagonal bin
            float bin_coord_x = target_bin_coords[0]; // left side
            float bin_coord_y = target_bin_coords[1]; // bottom side
            if (x>target_bin_coords[2])
                bin_coord_x = target_bin_coords[2]; // right side
            if (y>target_bin_coords[3])
                bin_coord_y = target_bin_coords[3]; // top side
            float pointCoord[2] = {bin_coord_x, bin_coord_y};
            distance_to_bin = calculate2dDistanceToThePoint(pointCoord, i_v, d_coord, n_coords);
        }
        
        // cout << "i_v: " << i_v << "; distance_to_bin: " << distance_to_bin << "; maxdistsq: " << maxdistsq << endl;
        
        if (distance_to_bin<maxdistsq){
            output_indices[n_output_vertices[0]] = i_v;
            n_output_vertices[0] += 1;
        }
    }
    
    return;
}

__device__
void produce_all_possible_pairs(const size_t index_x, const size_t index_y, 
                                const size_t n_bins_x, const size_t n_bins_y,
                                int* out_x, int* out_y){
    // TODO improve it
    int max_radius = (n_bins_x > n_bins_y) ? n_bins_x : n_bins_y;

    int counter = 0;

    int index = 0;
    int stride = 1;
    for (int radius=1 + index; radius<max_radius; radius += stride){
        int new_index_x = index_x-radius;
        if (new_index_x>=0 && new_index_x<n_bins_x){
            int tmp_radius = -radius;
            while (tmp_radius<=radius){
                int new_index_y = index_y+tmp_radius;
                if (new_index_y>=0 && new_index_y<n_bins_y){
                    out_x[counter] = new_index_x;
                    out_y[counter] = new_index_y;
                    counter++;
                }
                tmp_radius++;
            }
        }
        new_index_x = index_x+radius;
        if (new_index_x>=0 && new_index_x<n_bins_x){
            int tmp_radius = -radius;
            while (tmp_radius<=radius){
                int new_index_y = index_y+tmp_radius;
                if (new_index_y>=0 && new_index_y<n_bins_y){
                    out_x[counter] = new_index_x;
                    out_y[counter] = new_index_y;
                    counter++;
                }
                tmp_radius++;
            }
        }

        int new_index_y = index_y-radius;
        if (new_index_y>=0 && new_index_y<n_bins_y){
            int tmp_radius = -radius+1;
            while (tmp_radius<=(radius-1)){
                int new_index_x = index_x+tmp_radius;
                if (new_index_x>=0 && new_index_x<n_bins_x){
                    out_x[counter] = new_index_x;
                    out_y[counter] = new_index_y;
                    counter++;
                }
                tmp_radius++;
            }
        }
        new_index_y = index_y+radius;
        if (new_index_y>=0 && new_index_y<n_bins_y){
            int tmp_radius = -radius+1;
            while (tmp_radius<=(radius-1)){
                int new_index_x = index_x+tmp_radius;
                if (new_index_x>=0 && new_index_x<n_bins_x){
                    out_x[counter] = new_index_x;
                    out_y[counter] = new_index_y;
                    counter++;
                }
                tmp_radius++;
            }
        }
    }
}


__global__ 
void main_func(
        float* d_dist, // distance matrix
        int* d_indices, // indices matrix which corresponds to distance one
        const float *coords_2d_bins,
        const size_t* indices_of_vertices_in_bin,
        const size_t start_vert,
        const size_t end_vert,
        const int* n_vertices_in_bin,
        const float *d_coord,
        const size_t n_coords, // number of dimentions
        const size_t n_neigh, // number of neighbours
        const size_t n_bins_x,
        const size_t n_bins_y){


    const size_t iBinX = threadIdx.x/n_bins_x;
    const size_t iBinY = threadIdx.x - iBinX*n_bins_x;

    printf("main_func:\tthreadIdx.x: %d\n", threadIdx.x);
    printf("main_func:\tiBinX:%d;\tiBinY:%d\n", iBinX, iBinY);

    // const size_t iBinX =  blockIdx.x * blockDim.x + threadIdx.x;
    if(iBinX >= n_bins_x)
        return;
    // const size_t iBinY =  blockIdx.y * blockDim.y + threadIdx.y;
    if(iBinY >= n_bins_y)
        return;

    printf("pass\n");


    // const size_t iBinX =  blockIdx.x * blockDim.x + threadIdx.x;
    // if(iBinX >= n_bins_x)
    //     return;
    // const size_t iBinY =  blockIdx.y * blockDim.y + threadIdx.y;
    // if(iBinY >= n_bins_y)
    //     return;


    float binCoords[4];
    int counters[2];

    int tmp_indices[500000];
    int tmp_indices2[500000];

    bool bin_counter[1024];
    int loop_bins_x[1024];
    int loop_bins_y[1024];

    size_t bin_index = I2D(iBinX, iBinY, n_bins_y);
    for (int i=0; i<4; i++){
        binCoords[i] = coords_2d_bins[4*bin_index + i];
    }

    // printf("iBinX: %d;\tiBinY:%d\n", iBinX, iBinY);


    // prepare indices input array
    counters[0] = 0;

    gpu::prepare_indices(start_vert, end_vert, bin_index, indices_of_vertices_in_bin, counters, tmp_indices);

    gpu::findNeighbours(tmp_indices, // vertices for which we want to find neighbours in the targe phase-space bin
        n_vertices_in_bin[bin_index], // size of the first input array
        bin_index, // index of the newly added bin
        indices_of_vertices_in_bin, // index_map_to_bins[i_v] -> bin number to which vertex belong
        n_vertices_in_bin,
        d_coord,
        start_vert,
        end_vert,
        n_coords, // number of dimentions
        n_neigh, // number of neighbours
        d_dist,
        d_indices,
        -1.0
        );


    gpu::set_defaults_device(bin_counter, n_bins_x*n_bins_y, false);
    bin_counter[bin_index] = true;

    gpu::produce_all_possible_pairs(iBinX, iBinY, 
                    n_bins_x, n_bins_y,
                    loop_bins_x,loop_bins_y);

    counters[1] = 0;
    while (counters[1]<n_bins_x*n_bins_y-1){
        size_t iBinX2 = loop_bins_x[counters[1]];
        size_t iBinY2 = loop_bins_y[counters[1]];
        counters[1] += 1;
        
        size_t bin_index2 = I2D(iBinX2, iBinY2, n_bins_y);
        if (bin_counter[bin_index2]==true)
            continue;
        for (int i=0; i<4; i++){
            binCoords[i] = coords_2d_bins[4*bin_index2 + i];
        }

        counters[0] = 0;
        gpu::calculate2dDistanceToTheBinEdges(
            tmp_indices2, // out, not finished vertices 
            counters, // out, # of not finished vertices
            tmp_indices, // in, vertices which belong to bin #bin_index
            n_vertices_in_bin[bin_index], // in, n_verts in bin #bin_index
            binCoords, // {x1, y1, x2, y2} coords of the target bin
            d_coord,
            n_coords, 
            n_neigh, 
            d_dist, 
            d_indices, 
            -1.0
            // tmp_ndices2
            );

        if (counters[0]>0){
            // find neighbours
            gpu::findNeighbours(
                tmp_indices2, // in, vertices to search neighbours for
                counters[0], // in, n_verts
                bin_index2, // in, index of the newly added bin
                indices_of_vertices_in_bin, // index_map_to_bins[i_v] -> bin number to which vertex belong
                n_vertices_in_bin,
                d_coord,
                start_vert,
                end_vert,
                n_coords, // number of dimentions
                n_neigh, // number of neighbours
                d_dist,
                d_indices,
                -1.0
                );
        } // if (counters[0]>0)
    } // counter[1]
}

} // gpu namespace

typedef Eigen::GpuDevice GPUDevice;


template <typename dummy>
struct New3KnnOpFunctor<GPUDevice, dummy> {
    void operator()(const GPUDevice& d,

            const float *d_coord,
            const int* d_row_splits,
            int *d_indices,
            // int *_d_indices,
            float *d_dist,

            const int n_vert,
            const int n_neigh,
            const int n_coords,

            const int n_rs,
            const bool tf_compat,
            const float max_radius,
            const size_t n_bins_x,
            const size_t n_bins_y) {
        
        printf("SASHA: RUN CUDA KERNEL ON GPU!!!\n");
        
        // *******************************************
        // STEP 1: memmory allocation and set defaults
        // *******************************************
        int numBlocks_vert = (n_vert + BLOCK_SIZE - 1) / BLOCK_SIZE;
        // I am not sure if splits are working properly... TODO to test it
        // const size_t start_vert = d_row_splits[0];
        // const size_t end_vert = d_row_splits[1];
        const size_t start_vert = 0; // it's OK to create variables if one pass it as a copy
        const size_t end_vert = n_vert;

        float* coords_2d_bins; // one need to do cuda memory allocation for arrays!
        cudaMallocManaged(&coords_2d_bins, 4*n_bins_x*n_bins_y*sizeof(float));
        int* n_vertices_in_bin;
        cudaMallocManaged(&n_vertices_in_bin, n_bins_x*n_bins_y*sizeof(size_t));

        // printf("n_bins_x: %d;\tn_bins_y:%d\n", n_bins_x, n_bins_y);

        size_t *indices_of_vertices_in_bin;
        cudaMallocManaged(&indices_of_vertices_in_bin, (end_vert-start_vert)*sizeof(size_t));
        size_t *n_bins;
        cudaMallocManaged(&n_bins, 2*sizeof(size_t));
        n_bins[0] = n_bins_x;
        n_bins[1] = n_bins_y;
        //
        // TODO FIXME temporary do not search min and max, just use true value
        // to use some efficient existing solution
        float* min;
        float* max;
        cudaMallocManaged(&max, 2*sizeof(float));
        cudaMallocManaged(&min, 2*sizeof(float));
        min[0] = -0.0001;
        min[1] = -0.0001;
        max[0] = 1.0001;
        max[1] = 1.0001;


        gpu::set_defaults<<<numBlocks_vert,BLOCK_SIZE>>>(
                d_indices,
                d_dist,
                tf_compat,
                n_vert,
                n_neigh);

        // ************************************
        // STEP 2: divide vertices between bins
        // ************************************
        // printf("\n\n*** STEP 2: divide vertices between bins ***\n");

        gpu::set_defaults<<<numBlocks_vert,BLOCK_SIZE>>>(n_vertices_in_bin, n_bins_x*n_bins_y, 0);


        int numBlocks_slice = (n_bins_x*n_bins_y + BLOCK_SIZE - 1) / BLOCK_SIZE;
        gpu::get_bin_coords<<<numBlocks_slice, BLOCK_SIZE>>>(min, max, coords_2d_bins, n_bins_x, n_bins_y);
        cudaDeviceSynchronize();

        printf("DEBUG:\tcoords_2d_bins:\n");
        cpu::print_array(coords_2d_bins,0,4*n_bins_x*n_bins_y);

        gpu::constructPhaseSpaceBins<<<numBlocks_vert, BLOCK_SIZE>>>(
                d_coord, n_coords, start_vert, end_vert, n_bins, min, max,
                n_vertices_in_bin, indices_of_vertices_in_bin);
        cudaDeviceSynchronize();

        // ***********************
        // STEP 3: find neighbours
        // ***********************
        // printf("\n\n*** STEP 3: find neighbours ***\n");
        // float* binCoords;
        // cudaMallocManaged(&binCoords, 4*sizeof(float));
        // int *counters;
        // cudaMallocManaged(&counters, 2*sizeof(int));
        // int *tmp_indices;
        // cudaMallocManaged(&tmp_indices, (end_vert-start_vert)*sizeof(int));
        // int *tmp_indices2;
        // cudaMallocManaged(&tmp_indices2, (end_vert-start_vert)*sizeof(int));
        // bool *bin_counter;
        // cudaMallocManaged(&bin_counter, n_bins_x*n_bins_y*sizeof(bool));
        //
        // int *loop_bins_x;
        // cudaMallocManaged(&loop_bins_x, (n_bins_x*n_bins_y-1)*sizeof(int));
        // int *loop_bins_y;
        // cudaMallocManaged(&loop_bins_y, (n_bins_x*n_bins_y-1)*sizeof(int));

        // dim3 block_dim(128,1,1);
        // dim3 grid_dim(10,1,1); 
        // kernel<<<grid_dim,block_dim>>>(...);
        //  ==
        // threadIdx.x range from [0 ~ 128)
        // blockIdx.x range from [0 ~ 10)
        // blockDim.x equal to 128

        printf("DEBUG:\tn_bins_x: %d;\tn_bins_y:%d\n", n_bins_x, n_bins_y);
        printf("DEBUG2:\tn_bins_x: %d;\tn_bins_y:%d\n", n_bins_x, n_bins_y);
        
        gpu::main_func<<<numBlocks_slice, BLOCK_SIZE>>>(
            d_dist, 
            d_indices,
            coords_2d_bins,
            indices_of_vertices_in_bin,
            start_vert,
            end_vert,
            n_vertices_in_bin,
            d_coord,
            n_coords, 
            n_neigh, 
            n_bins_x,
            n_bins_y);
        cudaDeviceSynchronize();
        printf("DEBUG3:\tn_bins_x: %d;\tn_bins_y:%d\n", n_bins_x, n_bins_y);

        cudaFree(d_indices);
        cudaFree(coords_2d_bins);
        cudaFree(n_vertices_in_bin);
        cudaFree(indices_of_vertices_in_bin);
        cudaFree(min);
        cudaFree(max);
        cudaFree(n_bins);
        // cudaFree(binCoords);
        // cudaFree(counters);
        // cudaFree(tmp_indices);
        // cudaFree(tmp_indices2);
        // cudaFree(bin_counter);
        // cudaFree(loop_bins_x);
        // cudaFree(loop_bins_y);

        // FIXME do I need to do it in the end?
        // cudaDeviceSynchronize();

    }
};


template struct New3KnnOpFunctor<GPUDevice, int>;

}//functor
}//tensorflow


#endif  // GOOGLE_CUDA
