#include "benchmark_data.h"


exception_t* benchmark_data_analyze(benchmark_data_t* data) {
	char* function_name = "benchmark_data_init()";
	exception_t* exception;
	const long NANOSECONDS_PER_SECOND = 1000000000;
	double* times_elapsed;
	double temp;
	int i;

	// Validate parameters.
	#ifdef DEBUG_BENCHMARK_DATA
	if ( data == NULL ) {
		return exception_throw("benchmark_data was NULL.", function_name);
	}
	#endif

	// Allocate space for the times as double.
	times_elapsed = (double*)malloc(sizeof(double) * data->run_count);
	if ( times_elapsed == NULL ) {
		return exception_throw("Unable to allocate double array.", function_name);
	}

	// Convert each bit of data into a double.
	for ( i = 0; i < data->run_count; i++ ) {
		temp = (double)data->runs[i].time_elapsed.tv_nsec;
		temp /= NANOSECONDS_PER_SECOND;
		temp += (double)data->runs[i].time_elapsed.tv_sec;
		times_elapsed[i] = temp;
	}

	// Get the mean.
	exception = statistics_mean_sample_double(times_elapsed, data->run_count, &(data->mean_sample));
	if ( exception != NULL ) {
		return exception_append(exception, function_name);
	}

	// Get the harmonic mean.
	exception = statistics_mean_harmonic_double(times_elapsed, data->run_count, &(data->mean_harmonic));
	if ( exception != NULL ) {
		return exception_append(exception, function_name);
	}

	// Get the standard deviation.
	if ( data->run_count > 1 ) {
		exception = statistics_standard_deviation_double(times_elapsed, data->run_count, &(data->mean_sample), &(data->deviation));
		if ( exception != NULL ) {
			return exception_append(exception, function_name);
		}
	}
	else {
		data->deviation = 0.0f;
	}

	// Free the allocated times.
	free(times_elapsed);

	// Return success.
	return NULL;
}


exception_t* benchmark_data_init(benchmark_data_t* benchmark_data, int data_count) {
	char* function_name = "benchmark_data_init()";

	// Validate parameters.
	#ifdef DEBUG_BENCHMARK_DATA
	if ( benchmark_data == NULL ) {
		return exception_throw("benchmark_data was NULL.", function_name);
	}
	if ( data_count < 1 ) {
		return exception_throw("data_count was less than one.", function_name);
	}
	#endif

	// Allocate space for the benchmark data.
	benchmark_data->runs = (benchmark_run_t*)malloc(sizeof(benchmark_run_t) * data_count);
	if ( benchmark_data->runs == NULL ) {
		return exception_throw("Unable to allocate encryption data.", function_name);
	}

	// Set the number of benchmark data.
	benchmark_data->run_count = data_count;

	// Initialize the rest of the attributes.
	benchmark_data->mean_sample = 0.0f;
	benchmark_data->mean_harmonic = 0.0f;
	benchmark_data->deviation = 0.0f;

	// Return success.
	return NULL;
}


exception_t* benchmark_data_free(benchmark_data_t* benchmark_data) {
	#ifdef DEBUG_BENCHMARK_DATA
	char* function_name = "benchmark_data_free()";
	#endif

	// Validate parameters.
	#ifdef DEBUG_BENCHMARK_DATA
	if ( benchmark_data == NULL ) {
		return exception_throw("benchmark_data was NULL.", function_name);
	}
	#endif

	// Free the benchmark data.
	free(benchmark_data->runs);
	benchmark_data->runs = NULL;

	// Set the rest of the attributes.
	benchmark_data->run_count = -1;
	benchmark_data->mean_sample = -1.0f;
	benchmark_data->mean_harmonic = -1.0f;
	benchmark_data->deviation = -1.0f;

	// Return success.
	return NULL;
}
