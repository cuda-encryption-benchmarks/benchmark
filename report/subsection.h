
#ifndef subsection_H
#define subsection_H


#include "benchmark_data.h"
#include "../typedef.h"
#include "../ccc/ccc.h"
#include "../mode.h"


// The number of iterations to run for each subsection.
#define SUBSECTION_ITERATION_COUNT 10


typedef struct {
	// The mode that the algorithm is run in.
	enum mode mode;
	// The data gathered during encryption.
	benchmark_data_t data_encrypt[SUBSECTION_ITERATION_COUNT];
	// The data gathered during decryption.
	benchmark_data_t data_decrypt[SUBSECTION_ITERATION_COUNT];
} subsection_t;


/**	Initialize the specified subsection with the specified mode.
 *	@return	NULL on success, exception_t* on failure.
 */
exception_t* subsection_init(subsection_t* subsection, enum mode mode);


/**	Unitinialize the specified subsection.
 *	@return	NULL on success, exception_t* on failure.
 */
exception_t* subsection_free(subsection_t* subsection);


/**	Write the specified subsection to the specified pre-opened and writable file.
 *	@return	NULL on success, exception_t* on failure.
 */
exception_t* subsection_write(subsection_t* subsection, FILE* file);


#endif // subsection_H
