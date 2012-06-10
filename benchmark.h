
//! Macro to allow clock_getres() and related functionality.
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 199309L
#endif

#ifndef benchmark_H
#define benchmark_H


#include "ccc/ccc.h"
#include "aes.h"
#include "benchmark_data.h"
#include "file.h"
#include "key.h"
#include "serpent.h"
#include "twofish.h"
#include "typedef.h"


/*!	\brief Runs the benchmark with the specified parameters.
 *
 * 	\param[in]	key		The user-specified key.
 * 	\param[in]	input_filepath	Path from which to open the input file.
 * 	\param[in]	algorithm	Which algorithm to benchmark.
 * 	\param[in]	mode		How to run the specified algorithm.
 * 	\param[in]	encryption	Whether to run the encrypt or decrypt version of the algorithm.
 *	\param[out]	benchmark_run	The data from running the specified algorithm.
 *	\return	NULL on success, exception_t* on failure.
 */
exception_t* benchmark(key256_t* key, char* input_filepath, enum algorithm algorithm, enum mode mode, enum encryption encryption, benchmark_run_t* benchmark_run );


#endif //benchmark_H
