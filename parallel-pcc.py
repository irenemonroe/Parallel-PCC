/*
    *
    *   @title
    *       Compute PCC Matrix from any NxM Matrix.
    *
    *   @description
    *       This program reads NxM matrix from a
    *       CSV file, stores the matrix into an array, then spawns
    *       a kernel in GPU to compute the Pearson Correlation Coefficient (PCC) matrix. The PCC matrix
    *       is computed in a parallel fashion on the GPU.
    *
    *   @authors
    *       
    *       Bikash Jaiswal <bjjaiswal@gmail.com>
    *
    *	@date
    *	    July 22 2016
    *
    *
*/



import sys

import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

import time

import numpy as np

mod = SourceModule("""

#include <stdio.h>
#include <math.h>

__global__ void kernel( float *input, int *output, float thr ) {

    unsigned int bid_x = blockIdx.x;
    unsigned int bid_y = blockIdx.y;

    int rows = 5292;
    int cols = 107;

    unsigned int row_index_1 = (bid_x * cols);
    unsigned int row_index_2 = (bid_y * cols);

    float sum_x  = 0;
    float sum_y  = 0;
    float sum_xx = 0;
    float sum_yy = 0;
    float sum_xy = 0;

    unsigned int r1 = row_index_1;
    unsigned int r2 = row_index_2;

    for( int i = 0; i < cols  ; i++ ) {

        sum_x += input[r1];
        sum_y += input[r2];
        sum_xx += input[r1] * input[r1];
        sum_yy += input[r2] * input[r2];
        sum_xy += input[r1] * input[r2];

        r1++; r2++;

    }

    float pcc = 0.0;

    float num = ( cols * sum_xy ) - ( sum_x * sum_y );
    float den = (( cols * sum_xx ) - ( sum_x * sum_x )) * (( cols * sum_yy ) - ( sum_y * sum_y ));
    den = sqrt(den);

    pcc = (den == 0.0)? 0 : num/den;
    pcc = (pcc > 0) ? pcc : -pcc;

    pcc = (pcc > thr)? 1 : 0;

    //printf("> PCC : %f\\n", pcc);
    //printf(" > %f %f\\n", input[row_index_1 + 1], input[row_index_2 + 1]);

    unsigned int o_index = bid_x * rows + bid_y;

    if(bid_x == bid_y) {
        output[o_index] = 0.0;
    } else {
        output[o_index] = pcc;
    }

}

""")

d_input = np.genfromtxt(sys.argv[1], dtype=np.float32, delimiter=',')
#d_input = np.random.rand(5292,286).astype(np.float32)

#gpu_input = gpuarray.to_gpu(d_input)
input_shape = d_input.shape
gpu_input = cuda.mem_alloc(d_input.nbytes)
cuda.memcpy_htod(gpu_input, d_input)

rows = input_shape[0]
cols = input_shape[1]

thr = np.float32(sys.argv[3])

d_output = np.zeros(shape=(rows, rows), dtype=np.int32)
#gpu_output = gpuarray.to_gpu(d_output)
gpu_output = cuda.mem_alloc(d_output.nbytes)
cuda.memcpy_htod(gpu_output, d_output)

start = time.time()
kernel = mod.get_function("kernel")
kernel(gpu_input, gpu_output, thr, block=(1, 1, 1), grid=(rows, rows, 1))
finish = time.time()

d_result = np.empty_like(d_output)
cuda.memcpy_dtoh(d_result, gpu_output)
print("Time taken to compute: %0.3f ms\n" %((finish - start) * 1000.0 ))
np.savetxt(sys.argv[2], d_result, delimiter=",", fmt="%.0f")
