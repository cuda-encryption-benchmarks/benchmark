#include "statistics.h"


exception_t* statistics_mean_harmonic_double(double* values, int value_count, double* mean_output) {
	#ifdef DEBUG_STATISTICS
	char* function_name = "statistics_mean_harmonic_double()";
	#endif
	double mean;
	int i;

	// Validate
	#ifdef DEBUG_STATISTICS
	if ( vlaues == NULL ) {
		return exception_throw("values was NULL.", function_name);
	}
	else if ( value_count < 1 ) {
		return exception_throw("value_count was less than 1.", function_name);
	}
	else if ( mean_output == NULL ) {
		return exception_throw("mean_output was NULL.", function_name);
	}
	#endif

	// Calculate the mean.
	mean = 0.0f;
	for ( i = 0; i < value_count; i++ ) {
		mean += (1 / values[i]);
	}
	mean = ((double)value_count) / mean;

	// Assign output parameters.
	(*mean_output) = mean;

	// Return success.
	return NULL;
}


exception_t* statistics_mean_sample_double(double* values, int value_count, double* mean_output) {
	char* function_name = "statistics_mean_double()";
	double mean;
	int i;

	// Validate parameters.
	if ( values == NULL ) {
		return exception_throw("values was NULL.", function_name);
	}
	else if ( value_count < 1 ) {
		return exception_throw("value_count was less than 1.", function_name);
	}
	else if ( mean_output == NULL ) {
		return exception_throw("mean_output was NULL.", function_name);
	}

	// Calculate the mean.
	mean = 0.0f;
	for ( i = 0; i < value_count; i++ ) {
		mean += values[i];
	}
	mean /= value_count;

	// Assign output parameters.
	(*mean_output) = mean;

	// Return success.
	return NULL;
}


// s = sqrt( (1)/(n - 1) * sum_(i=1)^n((x_i - x_mean)^2) )
exception_t* statistics_standard_deviation_double(double* values, int value_count, double* mean_optional, double* standard_deviation) {
	char* function_name = "statistics_standard_deviation_double()";
	exception_t* exception;
	double mean;
	double temp;
	int i;

	// Validate parameters.
	if ( values == NULL ) {
		return exception_throw("values was NULL.", function_name);
	}
	else if ( value_count <= 1 ) {
		return exception_throw("Must have more than 1 value.", function_name);
	}
	else if ( standard_deviation == NULL ) {
		return exception_throw("standard_deviation was NULL.", function_name);
	}

	// Calculate the mean.
	if ( mean_optional == NULL ) {
		exception = statistics_mean_sample_double(values, value_count, &mean);
		if ( exception != NULL ) {
			return exception_append(exception, function_name);
		}
	}
	else {
		mean = (*mean_optional);
	}

	// Subtract each value from the mean and sum it.
	temp = 0.0f;
	for ( i = 0; i < value_count; i++ ) {
		temp += (values[i] - mean) * (values[i] - mean);
		//fprintf(stderr, "Temp: %f.\n", temp);
	}

	// Finish off the math.
	temp /= (value_count - 1);
	temp = sqrt(temp);

	//fprintf(stderr, "Standard deviation: %f.\n", temp);

	// Assign output parameters.
	(*standard_deviation) = temp;

	// Return success.
	return NULL;
}

