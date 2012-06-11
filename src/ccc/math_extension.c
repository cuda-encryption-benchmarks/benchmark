/*
 * Original Author: Wade Cline (clinew)
 * File: math_extension.c
 * Created: 2011 Late October/Early November by clinew
 * Last Modified: 2012 February 19, by clinew
 *
 * See: "math_extension.h".
 *
 */


#include "math_extension.h"


exception_t* math_dot_product( double* row_vector, double* column_vector, \
        int vector_lengths, double* dot_product ) {

	char* function_name = "math_dot_product()";
	int i = -1;

	// Validate row_vector.
	if ( row_vector == NULL ) {
		return exception_throw( "Unable to multiply a NULL pointer.", \
			function_name );
	}

	// Validate column_vector.
	if ( column_vector == NULL ) {
		return exception_throw( "Unable to multiply a NULL pointer.", function_name );
	}

	// Validate vector_lengths (as best possible).
	if ( vector_lengths < 1 ) {
		return exception_throw( "Vector length must be great than " \
			"zero.", function_name );
	}

	// Validate dot_product.
	if ( dot_product == NULL ) {
		return exception_throw( "Unable to assign a value to a " \
			"NULL pointer.", function_name );
	}

	// Initialize the dot product.
	*dot_product = 0.0;

	// Take the dot product of the two vectors.
	for ( i = 0; i < vector_lengths; i++ ) {
		*dot_product += row_vector[i] * column_vector[i];
		fprintf( stdout, "Dot product at iteration %i: %f\n", i, *dot_product );
	} 

	// Return success.
	return NULL;
}


exception_t* math_simpsons_rule(long double(*f)(long double), long double upper_bound, long double lower_bound, int subdivisions, long double* approximation) {
	char* function_name = "math_simpsons_rule()";
	long double width = 0xdeadbeef;
	int i = 0xdeadbeef;

	// Validate node count.
	if ( subdivisions < 1 ) {
		return exception_throw("Node count must be greater than zero.",
			function_name);
	}
	if ( subdivisions % 2 != 0 ) {
		return exception_throw("Unsure how to evaluate uneven node count.",
			function_name);
	}

	// Validate bounds.
	if ( lower_bound >= upper_bound) {
		return exception_throw("Lower bound must be lower than upper bound.",
			function_name);
	}

	// Validate output parameter.
	if ( approximation == NULL ) {
		return exception_throw("Unable to assign sum to a NULL value,",
			function_name);
	}
	
	// Calculate the width of each node.
	width = (upper_bound - lower_bound) / ((long double)subdivisions);

	// Sum each approximation. Note greater or equals to.
	// S_n(f) = (h/3)[f(x_0) + 4f(x_1) + 2f(x_2) + 4f(x_3) + 2f(x_4) + ... + 2f(x_{n-2}) + 4f(x_{n-1}) + f(x_n)].
	*approximation = 0.0;
	for ( i = 0; i <= subdivisions; i++ ) {
		long double multiplier = 0xdeadbeef;
		long double x_i = 0xdeadbeef;

		// Calculate node point.
		x_i = lower_bound + (i * width);

		// Calculate multiplier for this node point.
		if ( i % subdivisions == 0 ) {
			multiplier = 1.0;
		}
		else if ( i % 2 == 0 ) {
			multiplier = 2.0;
		}
		else { 
			multiplier = 4.0;
		}

		// Sum the approximation at each point x_i.
		*approximation += multiplier * f(x_i);
	}
	*approximation *= (width/3.0);

	// Return success.
	return NULL;
}


exception_t* math_trapezoidal_rule(long double (*f)(long double), long double upper_bound, long double lower_bound, int subintervals, long double* approximation) {
	char* function_name = "math_trapezoidal_rule()";
	long double width = 0xdeadbeef; //Normally denoted as "h" by mathematicians.
	int i = 0xdeadbeef;

	// Validate subinterval count.
	if ( subintervals < 1 ) {
		return exception_throw("Subinterval count must be greater than zero.",
			function_name);
	}

	// Validate bounds.
	if ( lower_bound >= upper_bound ) {
		return exception_throw("Lower bound must be lower than upper bound.",
			function_name);
	}

	// Validate output pointer.
	if ( approximation == NULL ) {
		return exception_throw("Unable to assign sum to a NULL value.",
			function_name);
	}
	
	// Calculate the width of each subinterval.
	width = (upper_bound - lower_bound) / ((long double)subintervals);

	// Sum each approximation. Note greater or equals to.
	// T_n(f) = h[(1/2)f(x_0) + f(x_1) + f(x_2) + ... + f(x_{n-1}) + (1/2)f(x_n)].
	*approximation = 0.0;
	for ( i = 0; i <= subintervals; i++ ) {
		long double multiplier = 0xdeadbeef;
		long double x_i = 0xdeadbeef;

		// Calculate point.
		x_i = lower_bound + (i * width);

		// Calculate multiplier (different for first and last iterations).
		if ( i % subintervals == 0 ) {
			multiplier = 1.0/2.0;
		}
		else {
			multiplier = 1.0;
		}

		// Sum the approximation at each point x_i.
		*approximation += width * multiplier * f(x_i);
	}
	
	// Return success.
	return NULL;
}
