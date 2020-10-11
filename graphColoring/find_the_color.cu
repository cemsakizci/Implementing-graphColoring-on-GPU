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

#include "find_the_color.h"

__global__ void find_the_color(Vertex *neighborsForAllVertices, int *neighborSizeArray, int current_phase, int *colors_found) {


	if(colors_found[blockIdx.x] == -1) {
		int starting_index_for_neighbors = 0; // showing the starting index of the specific neighbor set which differs for each block.
		int local_blockId = blockIdx.x;	
		// After the while loop finishes executing, each thread of the specific block will know the starting index of its specific neighbor set.
		while(local_blockId > 0) {
			starting_index_for_neighbors += neighborSizeArray[local_blockId-1];
			local_blockId--;
		}

		const int thread_number = 32;
		__shared__ long long int registers_pointer[thread_number][4];
		long long int registers[4] = {0, 0, 0, 0}; // local array storing 4 64-bit integer registers.

		// iterating all the neigbors of each vertex.
		for(int i = starting_index_for_neighbors + threadIdx.x; i < starting_index_for_neighbors + neighborSizeArray[blockIdx.x]; i += blockDim.x) {
				
			int color = neighborsForAllVertices[i].color; // get the color of the neighbor.

			if((color >= 256*current_phase+1) && (color <= 256*(current_phase+1))) {

				int register_color_range = (color-1) - current_phase*256; // which register color range this color fits in.
				int left_shift_amount = register_color_range % 64; // find how many left shifts are required.
				long long int mask = 1LL << left_shift_amount; // apply the left shift.  
				registers[register_color_range / 64] = registers[register_color_range / 64] | mask; // set the associated color bit by ORing the associated register.
			}
		}
		// updating the shared memory by thread's local registers array. 
		for(int i=0; i<4; i++) {
			registers_pointer[threadIdx.x][i] = registers[i];
		}
		__syncthreads(); // wait other threads to finish.
		
		// ORing all 256-bit bitflags and flipping each bit.
		if(threadIdx.x == 0) {
			long long int ORedregisters[4] = {0, 0, 0, 0};
			// Only the th0 will have the correct 256-bit bitflags since its local "ORedregisters" array is ORed by "registers_pointer" array.
			for(int j=0; j<4; j++) {
				for(int i=0; i<blockDim.x; i++) {
					ORedregisters[j] = ORedregisters[j] | registers_pointer[i][j];
				}
				ORedregisters[j] = ~ORedregisters[j]; //flipping all the bits. 
			}

			int color_to_be_used = -1;
			// finding the least significant bit set to 1.
			for(int i=0; i<4; i++) {
				int bit_position = __ffsll(ORedregisters[i]);
				if(bit_position != 0) {
					color_to_be_used = (i*64 + bit_position) + current_phase*256;
					break;
				}
			}

			// In case the color remains as -1, which is not in between the color range of the current phase, go to the next phase to find the appropriate color.
			*(colors_found + blockIdx.x) = color_to_be_used;
		}
	}
	
		
}
