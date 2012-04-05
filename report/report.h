
// Macro to allow clock_getres() and related functionality.
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 199309L
#endif

#ifndef report_H
#define report_H


#include <errno.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#include "section.h"
#include "../ccc/ccc.h"
#include "../typedef.h"


// The number of sections in the report.
// Section initialization will need to be hardcoded
// in the init function.
#define REPORT_SECTION_COUNT 1
// The section of the report for the Serpent algorithm.
#define REPORT_SECTION_SERPENT 0


// A structure that represents the report for the CUDA Benchmarking program.
typedef struct {
	// An array containing the sections of the report.
	section_t sections[REPORT_SECTION_COUNT];
} report_t;


/**	Initializes the members of the specified report_t.
 *	@return	NULL on success, exception_t* on failure.
 */
exception_t* report_init(report_t* report);


/**	Uninitializes the members of the specified report_t.
 *	@return NULL on success, exception_t* on failure.
 */
exception_t* report_free(report_t* report);


/**	Write the specified report to the specified filepath.
 *	@return	NULL on success, exception_t* on failure.
 */
exception_t* report_write(report_t* report);


/**	Private function of report_write() that creates the appropriate directory structure for the specified report.
 *	@out	report_filepath: Modifies the specified array of characters to contain the filepath for the specified report.
 *		The array should be at least 40 characters long.
 *	@return	NULL on success, exception_t* on failure.
 */
exception_t* report_write_create_directories(report_t* report, char* report_filepath);


#endif // report_H
