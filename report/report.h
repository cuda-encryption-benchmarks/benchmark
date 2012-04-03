
#ifndef report_H
#define report_H


#include "section.h"
#include "../ccc/ccc.h"


// The number of sections in the report.
// Section initialization will need to be hardcoded
// in the init function.
#define REPORT_SECTION_COUNT 1


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
exception_t* report_write(report_t* report, char* filepath);


#endif // report_H
