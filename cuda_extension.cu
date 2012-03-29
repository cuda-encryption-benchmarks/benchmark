#include "cuda_extension.h"


int cuda_get_block_and_thread_count_max(int device_number, int registers_per_thread, int* block_count, int* thread_count) {
	cudaDeviceProp device_properties;
	cudaError_t cuda_error;
	int thread_count_local;

	// Validate parameters.
	if ( block_count == NULL ) {
		fprintf(stderr, "block_count was NULL.\n");
		return -1;
	}
	if ( thread_count == NULL ) {
		fprintf(stderr, "thread_count was NULL.\n");
		return -1;
	}

	// Get device properties.
	cuda_error = cudaGetDeviceProperties( &device_properties, device_number );
	if ( cuda_error != cudaSuccess ) {
		fprintf(stderr, "Unable to get device properties: %s.\n", cudaGetErrorString(cuda_error));
		return -1;
	}

	// Calculate maximum thread count.
	thread_count_local = device_properties.regsPerBlock / registers_per_thread;
	if ( thread_count_local > device_properties.maxThreadsPerBlock ) {
		thread_count_local = device_properties.maxThreadsPerBlock;
	}

	// Set output parameters.
	(*block_count) = device_properties.multiProcessorCount;
	(*thread_count) = thread_count_local;

	// Return success.
	return 0;
}

