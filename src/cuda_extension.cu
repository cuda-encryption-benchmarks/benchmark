#include "cuda_extension.h"

#include <cuda.h>
#include <cuda_runtime_api.h>


extern "C"
int cuda_device_count(int* device_count) {
	cudaError_t cuda_error;

	// Validate parameters.
	if ( device_count == NULL ) {
		fprintf(stderr, "device_count was NULL.\n");
		return -1;
	}

	// Set device count.
	cuda_error = cudaGetDeviceCount(device_count);
	if ( cuda_error != cudaSuccess ) {
		fprintf(stderr, "ERROR getting device count: %s.\n", cudaGetErrorString(cuda_error));
		return -1;
	}

	// Return success.
	return 0;
}


extern "C"
int cuda_device_properties_report_write(FILE* file, int device_number) {
	cudaDeviceProp device_properties;
	cudaError_t cuda_error;
	int driver_version;
	int runtime_version;
	#if CUDART_VERSION >= 4000
	CUresult cu_result_error;
	int memory_clock;
	int memory_bus_width;
	int l2_cache_size;
	#endif

	// Validate parameters.
	if ( file == NULL ) {
		fprintf(stderr, "file was NULL.\n");
		return -1;
	}
	if ( device_number < 0 ) {
		fprintf(stderr, "device_number < 0.\n");
		return -1;
	}

	// Get device properties.
	cuda_error = cudaGetDeviceProperties(&device_properties, device_number);
	if ( cuda_error != cudaSuccess ) {
		fprintf(stderr, "Unable to get device #%i properties.\n", device_number);
		return -1;
	}

	// Get driver version.
	cuda_error = cudaDriverGetVersion(&driver_version);
	if ( cuda_error != cudaSuccess ) {
		fprintf(stderr, "Unable to get driver version.\n");
		return -1;
	}

	// Get runtime version.
	cuda_error = cudaRuntimeGetVersion(&runtime_version);
	if ( cuda_error != cudaSuccess ) {
		fprintf(stderr, "Unable to get runtime version.\n");
		return -1;
	}

	#if CUDART_VERSION >= 4000
	// Get memory clock.
	cu_result_error = cuDeviceGetAttribute( &memory_clock, CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, device_number);
	if ( cu_result_error != CUDA_SUCCESS ) {
		fprintf(stderr, "Unable to get memory clock.\n");
		return -1;
	}
	// Get memory bus width.
	cu_result_error = cuDeviceGetAttribute( &memory_bus_width, CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, device_number);
	if ( cu_result_error != CUDA_SUCCESS ) {
		fprintf(stderr, "Unable to get memory bus width.\n");
		return -1;
	}
	// Get L2 cache size.
	cu_result_error = cuDeviceGetAttribute( &l2_cache_size, CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, device_number);
	if ( cu_result_error != CUDA_SUCCESS ) {
		fprintf(stderr, "Unable to get L2 Cache Size.\n");
		return -1;
	}
	#endif

	// Write subsubsection header.
	fprintf(file, "\\subsubsection{%s}\n", device_properties.name);

	// Write table header.
	fprintf(file, "\\begin{figure}[H]\n" \
		"\\caption{Device \\#%i Properties}\n\\centering\n", device_number);

	// Write tabular head.
	fprintf(file, "\\begin{tabular}[c]{|p{7cm}|p{7cm}|}\n\\hline\n");

	// Write data.
	fprintf(file, "Name & %s \\\\\\hline\n", device_properties.name);
	fprintf(file, "Driver Version & %i.%i \\\\\\hline\n", driver_version/1000, (driver_version%100)/10);
	fprintf(file, "Runtime Version & %i.%i \\\\\\hline\n", runtime_version/1000, (runtime_version%100)/10);
	fprintf(file, "Capability Version & %i.%i \\\\\\hline\n", device_properties.major, device_properties.minor);
	fprintf(file, "Global Memory & %llu bytes \\\\\\hline\n", (unsigned long long)device_properties.totalGlobalMem);
	#if CUDART_VERSION >= 2000
	fprintf(file, "Multiprocessors & %i \\\\\\hline\n", device_properties.multiProcessorCount);
	// The necessary function does not appear to be provided with CUDA.
	//fprintf(file, "CUDA Cores/Multiprocessor & %i \\\\\n", ConvertSMVer2Cores(device_properties.major, device_properties.minor)); 
	#endif
	fprintf(file, "Clock Speed & %.2f GHz \\\\\\hline\n", device_properties.clockRate * 1e-3f);
	#if CUDART_VERSION >= 4000
	fprintf(file, "Memory Clock Rate & %.2f Mhz \\\\\\hline\n", memory_clock * 1e-3f);
	fprintf(file, "Memory Bus Width & %i-bit \\\\\\hline\n", memory_bus_width);
	if ( l2_cache_size != 0 ) {
		fprintf(file, "L2 Cache Size & %i bytes \\\\\\hline\n", l2_cache_size);
	}
	fprintf(file, "Max Texture Dimension Size (x,y,z) & 1D=(%i), 2D=(%i,%i), 3D=(%i,%i,%i) \\\\\\hline\n",
		device_properties.maxTexture1D, device_properties.maxTexture2D[0], device_properties.maxTexture2D[1],
		device_properties.maxTexture3D[0], device_properties.maxTexture3D[1], device_properties.maxTexture3D[2]);
	fprintf(file, "Max Layered Texture Size (dim) x layers & 1D=(%i) x %i, 2D=(%i,%i) x %i \\\\\\hline\n",
		device_properties.maxTexture1DLayered[0], device_properties.maxTexture1DLayered[1],
		device_properties.maxTexture2DLayered[0], device_properties.maxTexture2DLayered[1], device_properties.maxTexture2DLayered[2]);
	#endif
	#if defined (__LP64__) || defined(_LP64)
	fprintf(file, "Constant Memory & %lu bytes \\\\\\hline\n", device_properties.totalConstMem);
	fprintf(file, "Shared Memory per Block & %lu bytes \\\\\\hline\n", device_properties.sharedMemPerBlock);
	#else
	fprintf(file, "Constant Memory & %u bytes \\\\\\hline\n", device_properties.totalConstMem);
	fprintf(file, "Shared Memory per Block & %u bytes \\\\\\hline\n", device_properties.sharedMemPerBlock);
	#endif
	fprintf(file, "Registers per Block & %i \\\\\\hline\n", device_properties.regsPerBlock);
	fprintf(file, "Warp size & %i \\\\\\hline\n", device_properties.warpSize);
	fprintf(file, "Maximum number of threads per block & %i \\\\\\hline\n", device_properties.maxThreadsPerBlock);
	fprintf(file, "Maximum sizes of each dimension of a block & %i x %i x %i \\\\\\hline\n",
		device_properties.maxThreadsDim[0], device_properties.maxThreadsDim[1], device_properties.maxThreadsDim[2]);
	fprintf(file, "Maximum sizes of each dimension of a grid & %i x %i x %i \\\\\\hline\n",
		device_properties.maxGridSize[0], device_properties.maxGridSize[1], device_properties.maxGridSize[2]);
	#if defined (__LP64__) || defined(_LP64)
	fprintf(file, "Maximum memory pitch & %lu bytes \\\\\\hline\n", device_properties.memPitch);
	fprintf(file, "Texture alignment & %lu bytes \\\\\\hline\n", device_properties.textureAlignment);
	#else
	fprintf(file, "Maximum memory pitch & %u bytes \\\\\\hline\n", device_properties.memPitch);
	fprintf(file, "Texture alignment & %u bytes \\\\\\hline\n", device_properties.textureAlignment);
	#endif
	#if CUDART_VERSION >= 4000
	fprintf(file, "Concurrent copy and execution & %s with %i copy engine(s) \\\\\\hline\n",
		(device_properties.deviceOverlap ? "Yes" : "No"), device_properties.asyncEngineCount);
	#else
	fprintf(file, "Concurrent copy and execution & %s \\\\\\hline\n", device_properties.deviceOverlap ? "Yes" : "No");
	#endif
	#if CUDART_VERSION >= 2020
	fprintf(file, "Run time limit on kernels & %s \\\\\\hline\n", device_properties.kernelExecTimeoutEnabled ? "Yes" : "No");
	fprintf(file, "Integrated GPU sharing Host Memory & %s \\\\\\hline\n", device_properties.integrated ? "Yes" : "No");
	fprintf(file, "Support host page-locked memory mapping & %s \\\\\\hline\n", device_properties.canMapHostMemory ? "Yes" : "No");
	#endif
	#if CUDART_VERSION >= 3000
	fprintf(file, "Concurrent kernel execution & %s \\\\\\hline\n", device_properties.concurrentKernels ? "Yes" : "No");
	fprintf(file, "Alignment requirement for Surfaces & %s \\\\\\hline\n", device_properties.surfaceAlignment ? "Yes" : "No");
	#endif
	#if CUDART_VERSION >= 3010
	fprintf(file, "Device has EEC support enabled & %s \\\\\\hline\n", device_properties.ECCEnabled ? "Yes" : "No");
	#endif
	#if CUDART_VERSION >= 3020
	fprintf(file, "Device is using TCC driver mode & %s \\\\\\hline\n", device_properties.tccDriver ? "Yes" : "No");
	#endif
	#if CUDART_VERSION >= 4000
	fprintf(file, "Device supports Unified Addressing (UVA) & %s \\\\\\hline\n", device_properties.unifiedAddressing ? "Yes" : "No");
	fprintf(file, "Device PCI Bus ID / PCI location ID & %i / %i \\\\\\hline\n",
		device_properties.pciBusID, device_properties.pciDeviceID);
	#endif
	#if CUDART_VERSION >= 2020
	fprintf(file, "Compute Mode & ");
	switch(device_properties.computeMode) {
	case 0:
		fprintf(file, "Default (multiple host threads can use ::cudaSetDevice() with device simultaneously)");
		break;
	case 1:
		fprintf(file, "Exclusive (only one host thread in one process is able to use ::cudaSetDevice() with this device)");
		break;
	case 2:
		fprintf(file, "Prohibited (no host thread can use ::cudaSetDevice() with this device)");
		break;
	case 3:
		fprintf(file, "Exclusive Process (many threads in one process is able to use ::cudaSetDevice() with this device)");
		break;
	default:
		fprintf(file, "Unknown");
	}
	fprintf(file, " \\\\\\hline\n");
	#endif

	// Write tabular and table tails. 
	fprintf(file, "\\end{tabular}\n\\end{figure}\n");
	
	return 0;
}


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

