#include "implementation.h"

__global__ void find_the_color(Vertex *neighbors, const int neighborSize, int current_phase, int *color) {

	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	const int thread_number = 2; // this will be 32(warp size) for large inputs.
	__shared__ long long int registers_pointer[thread_number][4];
	long long int registers[4] = {0, 0, 0, 0}; // local array storing 4 64-bit integer registers.

	// iterating all the neigbors of the vertex.
	for(int i = idx; i < neighborSize; i += thread_number) {
			
		int color = neighbors[i].color; // get the color of the neighbor.

		if((color >= 256*current_phase+1) && (color <= 256*(current_phase+1))) {

			int register_color_range = (color-1) - current_phase*256; // which register color range this color fits in.
			int left_shift_amount = register_color_range % 64; // find how many left shifts are required.
			long long int mask = 1 << left_shift_amount; // apply the left shift.  
			registers[register_color_range / 64] = registers[register_color_range / 64] | mask; // set the associated color bit by ORing the associated register.
		}
	}
	// updating the shared memory by thread's local registers array. 
	for(int i=0; i<4; i++) {
		registers_pointer[idx][i] = registers[i];
	}
	__syncthreads(); // wait other threads to finish.
	
	// ORing all 256-bit bitflags and flipping each bit.
	if(idx == 0) {
		long long int ORedregisters[4] = {0, 0, 0, 0};
		// Only the th0 will have the correct 256-bit bitflags since its local "ORedregisters" array is ORed by "registers_pointer" array.
		for(int j=0; j<4; j++) {
			for(int i=0; i<thread_number; i++) {
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
		*color = color_to_be_used;
	}	
		
}
