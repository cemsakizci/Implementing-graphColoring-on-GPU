#ifndef GRAPHCOLORING_GPU_CUH
#define GRAPHCOLORING_GPU_CUH

#include <cuda.h>
#include "cuda_runtime.h"
#include <string.h>
#include "find_the_color.h"
#include "common.h"

typedef unsigned int uint;
typedef double dtype;

/*! \brief Host function for the graph coloring project.
 *
 * Please give a brief summary here.
 *
 * \param[in] arr - Input array of length loc_n.
 * \param[in,out] csum - Cumulative sum array with length loc_n.
 * \param[in] N - length of the total array (?).
 * \param[in] loc_n - Local N, length of the input array.
 * \return void - The function, being a CUDA kernel, does not return anything.
 *  */
extern void graphColoring_GPU(int numberOfVertices, Vertex *vertices);
#endif