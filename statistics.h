
#ifndef statistics_H
#define statistics_H


#ifdef DEBUG
#define DEBUG_STATISTICS
#endif


#include <math.h>

#include "ccc/ccc.h"


/*!	\brief Takes an array of doubles and returns the harmonic mean.
 *
 * 	\param[in]	values		The values to calculate the harmonic mean from.
 * 	\param[in]	value_count	The number of values in values.
 *	\param[out]	mean		The harmonic mean of the specified values.
 *	\return	NULL on success, exception_t* on failure.
 */
exception_t* statistics_mean_harmonic_double(double* values, int value_count, double* mean);


/*!	\brief Takes an array of doubles and returns the sample mean.
 *
 * 	\param[in]	values		The values to calculate the sample mean from.
 * 	\param[in]	value_count	The number of values in values.
 *	\param[out]	mean		The sample mean of the specified values.
 *	\return	NULL on success, exception_t* on failure.
 */
exception_t* statistics_mean_sample_double(double* values, int value_count, double* mean);


/*!	\brief Takes an array of doubles and returns the standard deviation as a double.
 *	\note Some precision may be lost due to floating-point rounding errors.
 *
 * 	\param[in]	values			The values to calculate the standard deviation from.
 * 	\param[in]	value_count		The number of values in values.
 *	\param[in]	mean			Optional parameter that specified the mean of the specified values; NULL to ignore.
 *	\param[out]	standard_deviation	The standard deviation of the specified values.
 *	\return	NULL on success, exception_t* on failure.
 */
exception_t* statistics_standard_deviation_double(double* values, int value_count, double* mean, double* standard_deviation);


#endif //statistics_H
