#include <cuda_runtime.h>
#include "vertex.h"

__global__ void find_the_color(Vertex *neighborsForAllVertices, int *neighborSizeArray, int current_phase, int *colors_found);
