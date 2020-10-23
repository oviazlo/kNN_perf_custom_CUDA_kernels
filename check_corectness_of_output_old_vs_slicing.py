from DeepJetCore.training.gpuTools import DJCSetGPUs
DJCSetGPUs()

import tensorflow as tf
import numpy as np

import time
from compare_knn_outputs_op import CompareKnnOutputs
from select_knn_op import SelectKnn
from new3_knn_op import New3Knn as NewKnn
#  from new_knn_op import NewKnn as NewKnn
#  from new2_knn_op import New2Knn as NewKnn
from rknn_op import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

tf.debugging.set_log_device_placement(True)

def createData(nvert,ncoords):
    coords = tf.constant( np.random.rand(nvert,ncoords) ,dtype='float32')
    row_splits = tf.constant( [0,  nvert] ,dtype='int32')
    return coords, row_splits

def euclidean_squared(A, B):
    A = tf.expand_dims(A, axis = 1) #V x 1 x C
    B = tf.expand_dims(B, axis = 0) #1 x V x C
    return tf.reduce_sum((A-B)**2, axis=-1)

def selectNeighbours_TF(K, coords, row_splits, return_distances=False):

    with tf.GradientTape(persistent=True,watch_accessed_variables=True) as t_newop:
        t_newop.watch(coords)

        out_indices=[]
        out_dst=[]
        for i in range(row_splits.shape[0]-1):

            distance_matrix = euclidean_squared(coords[row_splits[i]:row_splits[i+1]], coords[row_splits[i]:row_splits[i+1]])
            ranked_distances, ranked_indices = tf.nn.top_k(-distance_matrix, K)
            ranked_indices += row_splits[i]
            out_indices.append(ranked_indices)
            out_dst.append(ranked_distances)

        if return_distances:

           idcs=tf.concat(out_indices,axis=0)[...,tf.newaxis]

           distances = tf.reduce_sum(
               (coords[:, tf.newaxis, :] - tf.gather_nd(coords,idcs)) ** 2,
               axis=-1)

    if return_distances:
        return (idcs, distances), t_newop
    return tf.concat(out_indices,axis=0), t_newop

def selectNeighbours_CUDA(K, coords, row_splits, return_distances=False):
    with tf.GradientTape(persistent=True,watch_accessed_variables=True) as t_newop:
        t_newop.watch(coords)
        out = SelectKnn(K = K, coords=coords,  row_splits=row_splits,max_radius=-1., tf_compatible=True)
    return out, t_newop


def selectNeighbours_NewKnnCPU(K, coords, row_splits, return_distances=False, _n_bins_x=16, _n_bins_y=16):
    with tf.GradientTape(persistent=True,watch_accessed_variables=True) as t_newop:
        t_newop.watch(coords)
        out = NewKnn(K = K, coords=coords,  row_splits=row_splits,max_radius=-1., tf_compatible=True, n_bins_x = _n_bins_x, n_bins_y = _n_bins_y)
    return out, t_newop

def compareTensors(inTensor1, inTensor2):
    with tf.GradientTape(persistent=True,watch_accessed_variables=True) as t_newop:
        t_newop.watch(inTensor1)
        out = CompareKnnOutputs(inTensor1, inTensor2)
    return out, t_newop


def calculateIndecies_TF(coords, row_splits, nNeighbours):
    ind_tf, _ = selectNeighbours_TF(nNeighbours, coords, row_splits, return_distances=True)
    ind_tf =tf.squeeze(ind_tf[0],axis=2)
    return ind_tf

def calculateIndecies_oldKernel(coords, row_splits, nNeighbours):
    ind_custom, _ = selectNeighbours_CUDA(nNeighbours, coords, row_splits, return_distances=False)
    ind_custom = ind_custom[0]
    return ind_custom

def calculateIndecies_newKernel(coords, row_splits, nNeighbours, __n_bins_x=16, __n_bins_y=16):
    ind_newKnn, _ =  selectNeighbours_NewKnnCPU(nNeighbours, coords, row_splits, return_distances=False, _n_bins_x = __n_bins_x, _n_bins_y = __n_bins_y)
    ind_newKnn = ind_newKnn[0]
    return ind_newKnn


#****** MAIN ******
N_VERTICIES = 50
N_NEIGHBOURS = 5
N_DIMS = 4

coords, row_splits = createData(N_VERTICIES, N_DIMS)
#  print("***COORDS***")
#  print(coords)

try:
    with tf.device('/device:GPU:0'):
        ind_newKnn = calculateIndecies_newKernel(coords, row_splits, N_NEIGHBOURS, 4, 4)
except RuntimeError as e:
      print(e)

print("***INDECIES, NEW_KNN IMPL:***")
print(ind_newKnn)


#  with tf.device('/CPU:0'):
#      ind_tf = calculateIndecies_TF(N_VERTICIES,N_NEIGHBOURS, N_DIMS)
#
#  with tf.device('/GPU:4'):
#      ind_custom, ind_newKnn = calculateIndecies(N_VERTICIES,N_NEIGHBOURS, N_DIMS)
#
#  print("***DISTANCES, NEW_KNN IMPL:***")
#  print(ind_newKnn[1])
#
#  ind_custom = ind_custom[0]
#  ind_newKnn = ind_newKnn[0]
#
#  print("***INDECIES, TF IMPL:***")
#  print(ind_tf)
#  print("***INDECIES, CUDA IMPL:***")
#  print(ind_custom)
#  print("***INDECIES, NEW_KNN IMPL:***")
#  print(ind_newKnn)
#
#  outTensor=compareTensors(ind_tf, ind_custom)
#  print("***COMPARISON TENSOR: TF vs CUDA***")
#  print(outTensor)
#
#  outTensor=compareTensors(ind_custom, ind_newKnn)
#  print("***COMPARISON TENSOR: OLD vs. SLICING ***")
#  print(outTensor)
