
// Macro to allow clock_getres() and related functionality.
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 199309L
#endif


#ifndef benchmark_data_H
#define benchmark_data_H


#ifdef DEBUG
#define DEBUG_BENCHMARK_DATA
#endif


#include <time.h>

#include "benchmark_run.h"
#include "ccc/ccc.h"
#include "statistics.h"


// Structure representing an abstraction for the data gathered 
// by multiple runs of a single benchmark.
typedef struct {
	// Each individual run of the benchmark.
	benchmark_run_t* runs;
	// The total number of runs of the benchmark.
	int run_count;
	// The sample mean of the benchmark times.
	double mean_sample;
	// The harmonic mean of the benchmark times.
	double mean_harmonic;
	// The sample standard deviation of the benchmark times.
	double deviation;
} benchmark_data_t;


/**	Analyze the individual runs within the structure and set its members appropriately.
 *	@return	NULL on success, exception_t* on failure.
 */
exception_t* benchmark_data_analyze(benchmark_data_t* benchmark_data);


/**	Initialize the specified benchmark_data_t.
 *	@return	NULL on success, exception_t* on failure.
 */
exception_t* benchmark_data_init(benchmark_data_t* benchmark_data, int data_count);


/**	Free the specified benchmark_data_t.
 *	@return	NULL on success, exception_t* on failure.
 */
exception_t* benchmark_data_free(benchmark_data_t* benchmark_data);


#endif // benchmark_data_H
