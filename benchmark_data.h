
// Macro to allow clock_getres() and related functionality.
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 199309L
#endif


#ifndef benchmark_data_H
#define benchmark_data_H


#include <time.h>


// Structure representing an abstraction for the data gathered 
// by a single run of the benchmark.
typedef struct {
	// The total time taken for the run.
	struct timespec time_elapsed;
	// The amount of global memory used for the block buffer.
	size_t buffer_size;
} benchmark_data_t;


#endif // benchmark_data_H
