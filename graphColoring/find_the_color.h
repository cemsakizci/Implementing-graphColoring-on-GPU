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
#ifndef FIND_THE_COLOR_H
#define FIND_THE_COLOR_H
#include <cuda_runtime.h>
#include "vertex.h"

/*! \brief An global function to find the color.
 *
 * BRIEF SUMMARY HERE.
 * 
 * \param[in,out] inMTX - General matrix which will be loaded in CSR storage format.
 * \param[in] inMTXname - The filename of the matrix that will be loaded.
 * \return bool - returns a boolean value about importing the matrix.
 *
 * This function tries to open and load the given filename into the given general matrix and returns a true value. If the operation fails, it returns false value.
 */
__global__ void find_the_color(Vertex *neighborsForAllVertices, int *neighborSizeArray, int current_phase, int *colors_found);
#endif