// Implementation of the CJP algorithm, proposed by Nguyen Quang Anh Pham and Rui Fan, for parallel graph coloring on GPUs.
// Copyright (C) 2020, Cem Sakızcı <sakizcicem@gmail.com>

// This file is part of Implementing-graphColoring-on-GPU.

// Implementing-graphColoring-on-GPU is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// Implementing-graphColoring-on-GPU is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with Implementing-graphColoring-on-GPU.  If not, see <http://www.gnu.org/licenses/>.

#include <cuda_runtime.h>
#include "vertex.h"

__global__ void find_the_color(Vertex *neighborsForAllVertices, int *neighborSizeArray, int current_phase, int *colors_found);
