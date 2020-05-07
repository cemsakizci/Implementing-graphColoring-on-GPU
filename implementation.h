#include <cuda_runtime.h>
#include "vertex.h"

__global__ void find_the_color(Vertex *neighbors, const int neighborSize, int current_phase, int *color);
