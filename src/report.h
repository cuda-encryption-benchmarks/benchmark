
//! Macro to allow clock_getres() and related functionality.
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 199309L
#endif
//! Open large files.
#define _LARGEFILE64_SOURCE
//! For lstat().
#define _BSD_SOURCE

#ifndef report_H
#define report_H


#ifdef DEBUG
#define DEBUG_REPORT
#endif


#include <errno.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include "cuda_extension.h"
#include "section.h"
#include "ccc/ccc.h"
#include "typedef.h"


//! The maximum length of the report basepath string.
#define REPORT_BASEPATH_LENGTH_MAX 50
//! The maximum length of the report filename.
#define REPORT_FILENAME_LENGTH_MAX 10
/*! \brief The number of sections in the report.
 *  \note Section initialization will need to be hardcoded
 *  	in the init function.
 */
#define REPORT_SECTION_COUNT 2
//! The section of the report for the Serpent algorithm.
#define REPORT_SECTION_SERPENT 0
//! The section of the report for the Twofish algorithm.
#define REPORT_SECTION_TWOFISH 1


/*! \brief A structure that represents the report for the CUDA Benchmarking program.
 */
typedef struct {
	//! An array containing the sections of the report.
	section_t sections[REPORT_SECTION_COUNT];
	//! The file for the report to write to.
	FILE* file;
	//! The base path for the report's documents.
	char basepath[REPORT_BASEPATH_LENGTH_MAX];
	//! The filepath for the file to benchmark.
	char* input_filepath;
	//! The stats for the file benchmarked.
	struct stat64 stats;
	//! The name of the report without a file extension.
	char filename[REPORT_FILENAME_LENGTH_MAX];
} report_t;


/*!	\brief Execute the specified report. This will gather all necessary data and may take some time.
 *
 *	\return	NULL on success, exception_t* on failure.
 */
exception_t* report_execute(report_t* report);


/*!	\brief Initializes the members of the specified report_t.
 *
 *	\return	NULL on success, exception_t* on failure.
 */
exception_t* report_init(report_t* report, char* input_filepath, int data_count);


/*!	\brief Private function of report_init() that creates the appropriate directory structure for the specified report and initializes
 *		report->basepath as a side-effect.
 *
 *	\param[out]	report_basepath	Modifies the specified array of characters to contain the filepath for the specified report.
 *			The array should be at least 40 characters long.
 *	\return	NULL on success, exception_t* on failure.
 */
exception_t* report_init_create_directories(report_t* report);


/*!	\brief Uninitializes the members of the specified report_t and frees any related resources.
 *
 *	\return NULL on success, exception_t* on failure.
 */
exception_t* report_free(report_t* report);


/*!	\brief Write the specified report to the specified filepath.
 *
 *	\return	NULL on success, exception_t* on failure.
 */
exception_t* report_write(report_t* report);


/*!	\brief Private function of report_write() that writes the appendix of the report.
 * 	\note Tables may flow off the page if too much data is gathered.
 * 	\todo Change implementation to use the {longtable}(sp?) package.
 *
 *	\return	NULL on success, exception_t* on failure.
 */
exception_t* report_write_appendix(report_t* report);


/*!	\brief Private function of report_write() that compiles a LaTeX document.
 *
 *	\return	NULL on success, exception_t* on failure.
 */
exception_t* report_write_compile_latex(report_t* report);


/*!	\brief Private function of report_write() that appends the means of obtaining/calculating data to the report.
 *
 *	\return	NULL on success, exception_t* on failure.
 */
exception_t* report_write_methodologies(report_t* report);


/*!	\brief Private function of report_write() that writes the results of the report.
 *
 *	\return	NULL on success, exception_t* on failure.
 */
exception_t* report_write_results(report_t* report);


/*!	\brief Private function of report_write_results() that writes the gains between the algorithms.
 * 	
 *	\return	NULL on success, exception_t* on failure.
 */
exception_t* report_write_results_gain(report_t* report);


/*!	\brief Private function of report_write() that appends system information to the report.
 *
 *	\return	NULL on success, exception_t* on failure.
 */
exception_t* report_write_system_information(report_t* report);


/*!	\brief Private function of report_write_system_information() that writes data from each CUDA-capable device to the report.
 *
 *	\return	NULL on success, exception_t* on failure.
 */
exception_t* report_write_system_information_cuda_devices(report_t* report);


/*!	\brief Private function of report_write_system_information() that writes the output of the `lshw` command.
 *	to the report and a file.
 * 	\warning If the `lshw` command is not available, then no system
 * 		information will be gathered.
 *
 *	\return	NULL on success, exception_t* on failure.
 */
exception_t* report_write_system_information_lshw(report_t* report);


#endif // report_H
