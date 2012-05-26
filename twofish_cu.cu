/* See "twofish.h" for legal information */

#include "twofish.h"

#include <cuda.h>
#include <cuda_runtime_api.h>


/**     Decrypt a single block on the device.
 */
__device__ void twofish_cuda_decrypt_block(block128_t* block);


/**     Decrypt the specified array of blocks through a CUDA kernel.
 */
__global__ void twofish_cuda_decrypt_blocks(block128_t* cuda_blocks);


/**     Encrypt a single block on the device.
 */
__device__ void twofish_cuda_encrypt_block(block128_t* block);


/**     Encrypt the specified array of blocks through a CUDA kernel.
 */
__global__ void twofish_cuda_encrypt_blocks(block128_t* cuda_blocks);


/**     Flip the bytes of the specified 32-bit unsigned integer.
 *      NOTE: Tried to make a global function for this but got
 *      "Error: External calls are not supported...".
 *      @return A 32-bit unsigned integer with the bytes mirrored.
 */
__device__ uint32_t twofish_mirror_bytes32(uint32_t x);


// Constant variables in CUDA must be declared with a static scope.
// Some variables are prefixed with the file name because of
// "duplicate global variable looked up by string name" errors.
__device__ __constant__ uint32_t l_key[40];
__device__ __constant__ uint32_t mk_tab[1024];
// The total number of blocks being decrypted by a single CUDA thread.
__device__ __constant__ int twofish_blocks_per_thread;
// The total number of blocks being decrypted in the entire CUDA kernel.
__device__ __constant__ int twofish_blocks_per_kernel;



extern "C"
inline int twofish_cuda_allocate_buffer(size_t free_global_memory, size_t total_global_memory, int block_count, int multiprocessor_count, int thread_count, block128_t** cuda_blocks, size_t* used_global_memory_output, int* blocks_per_kernel_output, int* blocks_per_thread_output, int* buffer_allocation_attempts_output) {
	cudaError_t cuda_error;
	int blocks_per_kernel;
	int blocks_per_thread;
	int buffer_allocation_attempts;

	// Validate parameters.
	#ifdef DEBUG_TWOFISH
	if ( free_global_memory < 1 ) {
		fprintf(stderr, "Free global memory less than 1.\n");
		return -1;
	}
	else if ( cuda_blocks == NULL ) {
		fprintf(stderr, "cuda_blocks was NULL.\n");
		return -1;
	}
	else if ( used_global_memory_output == NULL ) {
		fprintf(stderr, "used_global_memory_output was NULL.\n");
		return -1;
	}
	else if ( blocks_per_kernel_output == NULL ) {
		fprintf(stderr, "blocks_per_kernel_output was NULL.\n");
		return -1;
	}
	else if ( blocks_per_thread_output == NULL ) {
		fprintf(stderr, "blocks_per_thread_output was NULL.\n");
		return -1;
	}
	#endif

	// Try to allocate blocks.
	buffer_allocation_attempts = 1;
	while ( true ) {
		// Subtract a small portion of global memory.
		free_global_memory -= total_global_memory * TWOFISH_CUDA_MEMORY_MULTIPLIER;
		if ( free_global_memory <= 0 ) {
			fprintf(stderr, "No memory for blocks available.\n");
			return -1;
		}

		// Calculate number of blocks per thread.
		blocks_per_kernel = free_global_memory / sizeof(block128_t);
		if ( blocks_per_kernel > block_count ) {
			blocks_per_kernel = block_count;
		}
		blocks_per_thread = (blocks_per_kernel / multiprocessor_count) / thread_count;

		// Attempt to allocate memory on the GPU.
		cuda_error = cudaMalloc( (void**)cuda_blocks, (int)(sizeof(block128_t) * blocks_per_kernel) );
		if ( cuda_error == cudaSuccess ) { // Success! Exit this loop.
			break;
		}
		else if ( cuda_error != cudaErrorMemoryAllocation ) {
			fprintf(stderr, "Unable to malloc blocks: %s.\n", cudaGetErrorString(cuda_error));
			return -1;
		}

		buffer_allocation_attempts++;
	}

	// Assign output parameters.
	(*used_global_memory_output) = free_global_memory;
	(*blocks_per_kernel_output) = blocks_per_kernel;
	(*blocks_per_thread_output) = blocks_per_thread;
	if ( buffer_allocation_attempts_output != NULL ) {
		(*buffer_allocation_attempts_output) = buffer_allocation_attempts;
	}

	// Return success.
	return 0;
}


__device__ void twofish_cuda_encrypt_block(block128_t* block) {
	uint32_t t0, t1, blk[4];
	int i;

	// Input whitening.
	blk[0] = twofish_mirror_bytes32(block->x0) ^ l_key[0];
	blk[1] = twofish_mirror_bytes32(block->x1) ^ l_key[1];
	blk[2] = twofish_mirror_bytes32(block->x2) ^ l_key[2];
	blk[3] = twofish_mirror_bytes32(block->x3) ^ l_key[3];

	// Run the 8 Fiestel network cycles.
	for ( i = 0; i < 8; i++ ) {
		t1 = g1_fun(blk[1]); t0 = g0_fun(blk[0]);
		blk[2] = rotr_fixed(blk[2] ^ (t0 + t1 + l_key[4 * (i) + 8]), 1);
		blk[3] = rotl_fixed(blk[3], 1) ^ (t0 + 2 * t1 + l_key[4 * (i) + 9]);
		t1 = g1_fun(blk[3]); t0 = g0_fun(blk[2]);
		blk[0] = rotr_fixed(blk[0] ^ (t0 + t1 + l_key[4 * (i) + 10]), 1);
		blk[1] = rotl_fixed(blk[1], 1) ^ (t0 + 2 * t1 + l_key[4 * (i) + 11]);
	}
	/* "Too many resources request for launch"
	f_rnd(0); f_rnd(1); f_rnd(2); f_rnd(3);
	f_rnd(4); f_rnd(5); f_rnd(6); f_rnd(7);
	*/

	// Output whitening.
	block->x0 = twofish_mirror_bytes32(blk[2] ^ l_key[4]);
	block->x1 = twofish_mirror_bytes32(blk[3] ^ l_key[5]);
	block->x2 = twofish_mirror_bytes32(blk[0] ^ l_key[6]);
	block->x3 = twofish_mirror_bytes32(blk[1] ^ l_key[7]);
}

__global__ void twofish_cuda_encrypt_blocks(block128_t* cuda_blocks) {
	int index = (blockIdx.x * (blockDim.x * twofish_blocks_per_thread)) + threadIdx.x;
	int i;

	// Encrypt the minimal number of blocks.
	for ( i = 0; i < twofish_blocks_per_thread; i++ ) {
		// Encrypt the block.
		twofish_cuda_encrypt_block(&(cuda_blocks[index]));

		// Increment the index.
		index += blockDim.x;
	}

	// Encrypt the extra blocks that fall outside the minimal number of block.s
	index = (gridDim.x * (blockDim.x * twofish_blocks_per_thread)) + ((blockIdx.x * blockDim.x) + threadIdx.x); // (end of array) + (absolute thread #).
	if ( index < twofish_blocks_per_kernel ) {
		twofish_cuda_encrypt_block(&(cuda_blocks[index]));
	}
}


extern "C"
int twofish_cuda_encrypt_cu(twofish_instance_t* instance, block128_t* blocks, int block_count, size_t* buffer_size) {
	// Maximum total number of registers taken up by a single CUDA thread.
	// This variable will need to be manually calculated and updated if
	// the algorithm implementation changes (but if you know of a way
	// to proceedurally do this, please, feel free...).
	const int REGISTERS_PER_THREAD = 19;
	block128_t* cuda_blocks;
	cudaError_t cuda_error;
	size_t total_global_memory;
	size_t free_global_memory;
	int buffer_allocation_attempts;
	int kernel_invocation_attempts;
	int blocks_per_kernel;
	int blocks_per_thread;
	int count;
	int device_number;
	int multiprocessor_count;
	int temp;
	int thread_count;
	int i;

	// Validate parameters.
	#ifdef DEBUG_TWOFISH
	if ( instance == NULL ) {
		fprintf(stderr, "instance was NULL.\n");
		return -1;
	}
	else if ( blocks == NULL ) {
		fprintf(stderr, "blocks was NULL.\n");
		return -1;
	}
	else if ( block_count < 1 ) {
		fprintf(stderr, "block_count was less than 1.\n");
		return -1;
	}
	else if ( buffer_size == NULL ) {
		fprintf(stderr, "buffer_size was NULL.\n");
		return -1;
	}
	#endif

	// Get the number of devices.
	cuda_error = cudaGetDeviceCount( &count );
	if ( cuda_error != cudaSuccess ) {
		fprintf(stderr, "Unable to get device count: %s.\n", cudaGetErrorString(cuda_error));
		return -1;
	}
	else if ( count == 0 ) {
		fprintf(stderr, "No CUDA-capable devices found.\n");
		return -1;
	}

	// Calculate the number of multiprocessors and threads to launch.
	device_number = 0;
	if ( cuda_get_block_and_thread_count_max(device_number, REGISTERS_PER_THREAD, &multiprocessor_count, &thread_count) == -1 ) {
		fprintf(stderr, "Unable to get max thread count.\n");
		return -1;
	}

	// Copy l_key and mk_tab to constant memory.
	cuda_error = cudaMemcpyToSymbol("l_key", instance->l_key, sizeof(uint32_t) * 40);
	if ( cuda_error != cudaSuccess ) {
		fprintf(stderr, "Unable to copy l_key to constant memory: %s.\n", cudaGetErrorString(cuda_error));
		return -1;
	}
	cuda_error = cudaMemcpyToSymbol("mk_tab", instance->mk_tab, sizeof(uint32_t) * 1024);
	if ( cuda_error != cudaSuccess ) {
		fprintf(stderr, "Unable to copy mk_tab to constant memory: %s.\n", cudaGetErrorString(cuda_error));
		return -1;
	}

	// Calculate the amount of global memory available for blocks.
	cuda_error = cudaMemGetInfo(&free_global_memory, &total_global_memory);
	if ( cuda_error != cudaSuccess ) {
		fprintf(stderr, "Unable to get memory information: %s.\n", cudaGetErrorString(cuda_error));
		return -1;
	}

	// Allocate blocks buffer on the GPU.
	if ( twofish_cuda_allocate_buffer(free_global_memory, total_global_memory, block_count, multiprocessor_count, thread_count, &cuda_blocks, &free_global_memory, &blocks_per_kernel, &blocks_per_thread, &buffer_allocation_attempts) == -1 ) {
		fprintf(stderr, "Unable to allocate initial buffer.\n");
		return -1;
	}

	// Encrypt the blocks.
	i = 0;
	while (i < block_count) {
		// Corner case.
		if ( i + blocks_per_kernel > block_count ) {
			blocks_per_kernel = block_count - i;
			blocks_per_thread = blocks_per_kernel / multiprocessor_count / thread_count;
		}

		// Run the algorithm.
		kernel_invocation_attempts = 1;
		while ( true ) {
			// Move blocks to global memory.
			cuda_error = cudaMemcpy( cuda_blocks, &(blocks[i]), sizeof(block128_t) * blocks_per_kernel, cudaMemcpyHostToDevice );
			if ( cuda_error != cudaSuccess ) {
				fprintf(stderr, "Unable to memcopy blocks: %s.\n", cudaGetErrorString(cuda_error));
				return -1;
			}

			// Copy blocks per thread to constant memory.
			cuda_error = cudaMemcpyToSymbol( "twofish_blocks_per_thread", &blocks_per_thread, sizeof(int));
			if ( cuda_error != cudaSuccess ) {
				fprintf(stderr, "Unable to copy blocks_per_thread to constant memory: %s.\n", cudaGetErrorString(cuda_error));
				return -1;
			}

			// Copy blocks per kernel to constant memory.
			cuda_error = cudaMemcpyToSymbol( "twofish_blocks_per_kernel", &blocks_per_kernel, sizeof(int));
			if ( cuda_error != cudaSuccess ) {
				fprintf(stderr, "Unable to copy blocks_per_kernel to constant memory: %s.\n", cudaGetErrorString(cuda_error));
				return -1;
			}

			// Run encryption.
			twofish_cuda_encrypt_blocks<<<multiprocessor_count, thread_count>>>(cuda_blocks);
			cuda_error = cudaGetLastError();
			if ( cuda_error == cudaSuccess ) { // Successful run.
				break;
			}
			else if ( cuda_error != cudaErrorMemoryAllocation ) { // Unexpected error.
				fprintf(stderr, "ERROR invoking the kernel: %s, %i.\n", cudaGetErrorString(cuda_error), cuda_error);
				return -1;
			}

			// Free the old blocks buffer.
			cudaFree(cuda_blocks);
	
			// Allocate a new blocks buffer.
			if ( twofish_cuda_allocate_buffer(free_global_memory, total_global_memory, block_count, multiprocessor_count, thread_count, &cuda_blocks, &free_global_memory, &blocks_per_kernel, &blocks_per_thread, &temp) == -1 ) {
				fprintf(stderr, "Unable to reallocate blocks buffer.\n");
				return -1;
			}
			buffer_allocation_attempts += temp;

			kernel_invocation_attempts++;
		}

		// Get blocks from global memory.
		cuda_error = cudaMemcpy( &(blocks[i]), cuda_blocks, sizeof(block128_t) * blocks_per_kernel, cudaMemcpyDeviceToHost );
		if ( cuda_error != cudaSuccess ) {
			fprintf(stderr, "Unable to retrieve blocks: %s.\n", cudaGetErrorString(cuda_error));
			return -1;
		}

		// Increment i by the number of blocks processed.
		i += blocks_per_kernel;
	}

	// Free blocks from global memory.
	cudaFree(cuda_blocks);

	// Assign output parameters.
	(*buffer_size) = free_global_memory;

	// TODO: Makes this as function output.
	//fprintf(stderr, "Buffer allocation attempts: %i.\nKernel invocation attempts: %i.\n",
		//buffer_allocation_attempts, kernel_invocation_attempts);

	// Return success.
	return 0;
}


__device__ void twofish_cuda_decrypt_block(block128_t* block) {
	uint32_t t0, t1, blk[4];
	int i;

	// Input whitening.
	blk[0] = twofish_mirror_bytes32(block->x0) ^ l_key[4];
	blk[1] = twofish_mirror_bytes32(block->x1) ^ l_key[5];
	blk[2] = twofish_mirror_bytes32(block->x2) ^ l_key[6];
	blk[3] = twofish_mirror_bytes32(block->x3) ^ l_key[7];

	// Run the 8 Fiestel network cycles.
	for (i = 7; i >= 0; --i) {
		t1 = g1_fun(blk[1]); t0 = g0_fun(blk[0]);
		blk[2] = rotl_fixed(blk[2], 1) ^ (t0 + t1 + l_key[4 * (i) + 10]);
		blk[3] = rotr_fixed(blk[3] ^ (t0 + 2 * t1 + l_key[4 * (i) + 11]), 1);
		t1 = g1_fun(blk[3]); t0 = g0_fun(blk[2]);
		blk[0] = rotl_fixed(blk[0], 1) ^ (t0 + t1 + l_key[4 * (i) +  8]);
		blk[1] = rotr_fixed(blk[1] ^ (t0 + 2 * t1 + l_key[4 * (i) +  9]), 1);
	}
	/* "Too many resources request for launch"
	f_rnd(0); f_rnd(1); f_rnd(2); f_rnd(3);
	f_rnd(4); f_rnd(5); f_rnd(6); f_rnd(7);
	*/

	// Output whitening.
	block->x0 = twofish_mirror_bytes32(blk[2] ^ l_key[0]);
	block->x1 = twofish_mirror_bytes32(blk[3] ^ l_key[1]);
	block->x2 = twofish_mirror_bytes32(blk[0] ^ l_key[2]);
	block->x3 = twofish_mirror_bytes32(blk[1] ^ l_key[3]);
}


__global__ void twofish_cuda_decrypt_blocks(block128_t* cuda_blocks) {
	int index = (blockIdx.x * (blockDim.x * twofish_blocks_per_thread)) + threadIdx.x;
	int i;

	// Encrypt the minimal number of blocks.
	for ( i = 0; i < twofish_blocks_per_thread; i++ ) {
		// Encrypt the block.
		twofish_cuda_decrypt_block(&(cuda_blocks[index]));

		// Increment the index.
		index += blockDim.x;
	}

	// Encrypt the extra blocks that fall outside the minimal number of block.s
	index = (gridDim.x * (blockDim.x * twofish_blocks_per_thread)) + ((blockIdx.x * blockDim.x) + threadIdx.x); // (end of array) + (absolute thread #).
	if ( index < twofish_blocks_per_kernel ) {
		twofish_cuda_decrypt_block(&(cuda_blocks[index]));
	}
}


extern "C"
int twofish_cuda_decrypt_cu(twofish_instance_t* instance, block128_t* blocks, int block_count, size_t* buffer_size) {
	// Maximum total number of registers taken up by a single CUDA thread.
	// This variable will need to be manually calculated and updated if
	// the algorithm implementation changes (but if you know of a way
	// to proceedurally do this, please, feel free...).
	const int REGISTERS_PER_THREAD = 19;
	block128_t* cuda_blocks;
	cudaError_t cuda_error;
	size_t total_global_memory;
	size_t free_global_memory;
	int buffer_allocation_attempts;
	int kernel_invocation_attempts;
	int blocks_per_kernel;
	int blocks_per_thread;
	int count;
	int device_number;
	int multiprocessor_count;
	int temp;
	int thread_count;
	int i;

	// Validate parameters.
	#ifdef DEBUG_TWOFISH
	if ( instance == NULL ) {
		fprintf(stderr, "instance was NULL.\n");
		return -1;
	}
	else if ( blocks == NULL ) {
		fprintf(stderr, "blocks was NULL.\n");
		return -1;
	}
	else if ( block_count < 1 ) {
		fprintf(stderr, "block_count was less than 1.\n");
		return -1;
	}
	else if ( buffer_size == NULL ) {
		fprintf(stderr, "buffer_size was NULL.\n");
		return -1;
	}
	#endif

	// Get the number of devices.
	cuda_error = cudaGetDeviceCount( &count );
	if ( cuda_error != cudaSuccess ) {
		fprintf(stderr, "Unable to get device count: %s.\n", cudaGetErrorString(cuda_error));
		return -1;
	}
	else if ( count == 0 ) {
		fprintf(stderr, "No CUDA-capable devices found.\n");
		return -1;
	}

	// Calculate the number of multiprocessors and threads to launch.
	device_number = 0;
	if ( cuda_get_block_and_thread_count_max(device_number, REGISTERS_PER_THREAD, &multiprocessor_count, &thread_count) == -1 ) {
		fprintf(stderr, "Unable to get max thread count.\n");
		return -1;
	}

	// Copy l_key and mk_tab to constant memory.
	cuda_error = cudaMemcpyToSymbol("l_key", instance->l_key, sizeof(uint32_t) * 40);
	if ( cuda_error != cudaSuccess ) {
		fprintf(stderr, "Unable to copy l_key to constant memory: %s.\n", cudaGetErrorString(cuda_error));
		return -1;
	}
	cuda_error = cudaMemcpyToSymbol("mk_tab", instance->mk_tab, sizeof(uint32_t) * 1024);
	if ( cuda_error != cudaSuccess ) {
		fprintf(stderr, "Unable to copy mk_tab to constant memory: %s.\n", cudaGetErrorString(cuda_error));
		return -1;
	}

	// Calculate the amount of global memory available for blocks.
	cuda_error = cudaMemGetInfo(&free_global_memory, &total_global_memory);
	if ( cuda_error != cudaSuccess ) {
		fprintf(stderr, "Unable to get memory information: %s.\n", cudaGetErrorString(cuda_error));
		return -1;
	}

	// Allocate blocks buffer on the GPU.
	if ( twofish_cuda_allocate_buffer(free_global_memory, total_global_memory, block_count, multiprocessor_count, thread_count, &cuda_blocks, &free_global_memory, &blocks_per_kernel, &blocks_per_thread, &buffer_allocation_attempts) == -1 ) {
		fprintf(stderr, "Unable to allocate initial buffer.\n");
		return -1;
	}

	// Decrypt the blocks.
	i = 0;
	while (i < block_count) {
		// Corner case.
		if ( i + blocks_per_kernel > block_count ) {
			blocks_per_kernel = block_count - i;
			blocks_per_thread = blocks_per_kernel / multiprocessor_count / thread_count;
		}

		// Run the algorithm.
		kernel_invocation_attempts = 1;
		while ( true ) {
			// Move blocks to global memory.
			cuda_error = cudaMemcpy( cuda_blocks, &(blocks[i]), sizeof(block128_t) * blocks_per_kernel, cudaMemcpyHostToDevice );
			if ( cuda_error != cudaSuccess ) {
				fprintf(stderr, "Unable to memcopy blocks: %s.\n", cudaGetErrorString(cuda_error));
				return -1;
			}

			// Copy blocks per thread to constant memory.
			cuda_error = cudaMemcpyToSymbol( "twofish_blocks_per_thread", &blocks_per_thread, sizeof(int));
			if ( cuda_error != cudaSuccess ) {
				fprintf(stderr, "Unable to copy blocks_per_thread to constant memory: %s.\n", cudaGetErrorString(cuda_error));
				return -1;
			}

			// Copy blocks per kernel to constant memory.
			cuda_error = cudaMemcpyToSymbol( "twofish_blocks_per_kernel", &blocks_per_kernel, sizeof(int));
			if ( cuda_error != cudaSuccess ) {
				fprintf(stderr, "Unable to copy blocks_per_kernel to constant memory: %s.\n", cudaGetErrorString(cuda_error));
				return -1;
			}

			// Run decryption.
			twofish_cuda_decrypt_blocks<<<multiprocessor_count, thread_count>>>(cuda_blocks);
			cuda_error = cudaGetLastError();
			if ( cuda_error == cudaSuccess ) { // Successful run.
				break;
			}
			else if ( cuda_error != cudaErrorMemoryAllocation ) { // Unexpected error.
				fprintf(stderr, "ERROR invoking the kernel: %s, %i.\n", cudaGetErrorString(cuda_error), cuda_error);
				return -1;
			}

			// Free the old blocks buffer.
			cudaFree(cuda_blocks);
	
			// Allocate a new blocks buffer.
			if ( twofish_cuda_allocate_buffer(free_global_memory, total_global_memory, block_count, multiprocessor_count, thread_count, &cuda_blocks, &free_global_memory, &blocks_per_kernel, &blocks_per_thread, &temp) == -1 ) {
				fprintf(stderr, "Unable to reallocate blocks buffer.\n");
				return -1;
			}
			buffer_allocation_attempts += temp;

			kernel_invocation_attempts++;
		}

		// Get blocks from global memory.
		cuda_error = cudaMemcpy( &(blocks[i]), cuda_blocks, sizeof(block128_t) * blocks_per_kernel, cudaMemcpyDeviceToHost );
		if ( cuda_error != cudaSuccess ) {
			fprintf(stderr, "Unable to retrieve blocks: %s.\n", cudaGetErrorString(cuda_error));
			return -1;
		}

		// Increment i by the number of blocks processed.
		i += blocks_per_kernel;
	}

	// Free blocks from global memory.
	cudaFree(cuda_blocks);

	// Assign output parameters.
	(*buffer_size) = free_global_memory;

	// TODO: Makes this as function output.
	//fprintf(stderr, "Buffer allocation attempts: %i.\nKernel invocation attempts: %i.\n",
		//buffer_allocation_attempts, kernel_invocation_attempts);

	// Return success.
	return 0;
}


__device__ uint32_t twofish_mirror_bytes32(uint32_t x) {
	uint32_t out;

	// Change to Little Endian.
	out = (uint8_t) x;
	out <<= 8; out |= (uint8_t) (x >> 8);
	out <<= 8; out |= (uint8_t) (x >> 16);
	out = (out << 8) | (uint8_t) (x >> 24);

	// Return out.
	return out;
}
