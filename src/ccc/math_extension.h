/*
 * Original Author: Wade Cline (clinew)
 * File: math_extension.h
 * Created: 2011 Late October/Early November by clinew
 * Last Modified: 2012 February 19, by clinew
 *
 * Contains various utilities that may be useful for various math operations. Linking a program with these functions may require the math library ('-lm').
 */


#ifndef math_extension_H
#define math_extension_H


#include "exception.h"


/**	Multiplies the two double arrays together in order to get the dot
 *	product. Excessive values of vector_lengths may cause excessive
 *	rounding errors.
 *	@pre	vector_lengths represents the actual length of both vectors.
 *	@out	dot_product: The dot product of the two double arrays.
 *	@return	NULL on success, exception_t* on failure.
 */
exception_t* math_dot_product( double* row_vector, double* column_vector, \
	int vector_lengths, double* dot_product );


/**	Performs Simpson's Rule of integral approximation for the specified function f(x) given the specified upper and lower bounds, and the specified number of subdivisions.
 * 	@pre	subdivisions must be an even integer.
 *	@out	approximation: The result of the approximation using Simpson's Rule.
 *	@return	NULL on success, exception_t* on failure.
 */
exception_t* math_simpsons_rule(long double(*f)(long double), long double upper_bound, long double lower_bound, int subdivisions, long double* approximation);


/**	Performs the trapezoidal rule of integral approximation for the specified function f(x) given the specified upper and lower bounds, and the specified number of subintervals (n).
 *	@out	approximation: The result of the approximation using the trapezoidal rule.
 *	@return	NULL on success, exception_t* on failure.
 */
exception_t* math_trapezoidal_rule(long double (*f)(long double), long double upper_bound, long double lower_bound, int subintervals, long double* approximation);


#endif
