__global__ void find_the_color(Vertex *neighbors, const int neighborSize, int current_phase) {

	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	__shared__ long long int registers_pointer[32][4]; 
	long long int registers[4] = {0, 0, 0, 0}; // local array storing 4 64-bit integer registers.
	long long int ORedregisters[4] = {0, 0, 0, 0};

	// iterating all the neigbors of the vertex.
	for(int i = idx; i < neighborSize; i += 32) {
			
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
		for(int j=0; j<4; j++) {
			for(int i=0; i<32; i++) {
				ORedregisters[j] = ORedregisters[j] | registers_pointer[i][j];
			}
			ORedregisters[j] = ~ORedregisters[j]; // flipping all the bits.
		}
	}
	
	int color_to_be_used = -1;
	// finding the least significant bit set to 1.
	for(int i=0; i<4; i++) {
		int bit_position = __ffsll(ORedregisters[j]);
		if(bit_position != 0) {
			color_to_be_used = (i*64 + bit_position) + current_phase*256;
			break;
		}
	}

	if(color_to_be_used == -1) {
		// go to the next phase.
	}
	else {
		// color has been found.
	}
		
		
}
