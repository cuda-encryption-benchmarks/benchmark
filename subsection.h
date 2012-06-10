
//! Macro to allow clock_getres() and related functionality.
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 199309L
#endif


#ifndef subsection_H
#define subsection_H


#ifdef DEBUG
#define DEBUG_SUBSECTION
#endif


#include <ctype.h>


#include "benchmark_data.h"
#include "benchmark.h"
#include "typedef.h"
#include "ccc/ccc.h"
#include "mode.h"
#include "key.h"
#include "statistics.h"


/*! \brief Holds the data for a single encryption algorithm.
 * 	This includes the data from both the encrypt and decrypt
 * 	versions of the algorithm.
 */
typedef struct {
	//! The mode that the algorithm is run in.
	enum mode mode;
	//! The number of data entries.
	int data_count;
	//! The data gathered during encryption.
	benchmark_data_t data_encrypt;
	//! The data gathered during decryption.
	benchmark_data_t data_decrypt;
} subsection_t;


/*!	\brief Execute the data-gathering phase of the specified subsection. This could take some time.
 *
 *	\return	NULL on success, exception_t* on failure.
 */
exception_t* subsection_execute(subsection_t* subsection, key256_t* key, char* input_filepath, enum algorithm algorithm);


/*!	\brief Initialize the specified subsection with the specified mode.
 *
 *	\return	NULL on success, exception_t* on failure.
 */
exception_t* subsection_init(subsection_t* subsection, int data_count, enum mode mode);


/*!	\brief Unitinialize the specified subsection.
 *
 *	\return	NULL on success, exception_t* on failure.
 */
exception_t* subsection_free(subsection_t* subsection);


/*!	\brief Write the specified subsection to the specified pre-opened and writable file.
 *
 *	\return	NULL on success, exception_t* on failure.
 */
exception_t* subsection_write_appendix(subsection_t* subsection, FILE* file, enum algorithm algorithm);


/*!	\brief Private function of subsection_write_appendix() for writing CSV output.
 *
 *	\return	NULL on success, exception_t* on failure.
 */
exception_t* subsection_write_appendix_csv(subsection_t* subsection, enum algorithm algorithm);


/*!	\brief Private function of subsection_write_appendix_csv() for writing a single CSV file.
 *
 *	\return	NULL on success, exception_t* on failure.
 */
exception_t* subsection_write_appendix_csv_file(benchmark_data_t* data, int data_count, enum algorithm algorithm, enum mode mode, enum encryption encryption);


/*!	\brief Private function of subsection_write_appendix() for writing the LaTeX output.
 *
 *	\return	NULL on success, exception_t* on failure.
 */
exception_t* subsection_write_appendix_latex(subsection_t* subsection, FILE* file, enum algorithm algorithm);


/*!	\brief Private function of subsection_write_appendix_latex() that writes part of the subsection data in LaTeX.
 *
 *	\return	NULL on success, exception_t* on failure.
 */
exception_t* subsection_write_appendix_latex_data(FILE* file, benchmark_data_t* data, int data_count, enum algorithm algorithm, enum mode mode, enum encryption encryption);


/*!	\brief Private function of subsection_write_appendix_latex_data() that writes statistical information in LaTeX.
 * 	\deprecated This is now handled when writing the overall gains for the report.
 *
 *	\return	NULL on success, exception_t* on failure.
 */
exception_t* subsection_write_appendix_latex_data_statistics(FILE* file, benchmark_data_t* data, int data_count);


/*!	\brief Private function of subsection_write_appendix_latex_data() for writing data tables in LaTeX.
 * 
 *	\return	NULL on success, exception_t* on failure.
 */
exception_t* subsection_write_appendix_latex_data_table(FILE* file, benchmark_data_t* data, int data_count, enum algorithm algorithm, enum mode mode, enum encryption encryption);


/*!	\brief Write either the encryption or decryption summary results as a LaTeX tabular row.
 *
 *	\return	NULL on success, exception_t* on failure.
 */
exception_t* subsection_write_results_summary_table_row(subsection_t* subsection, FILE* file, off_t size, enum encryption encryption);


#endif // subsection_H
