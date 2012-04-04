#include "report.h"


exception_t* report_init(report_t* report) {
	char* function_name = "report_init()";
	exception_t* exception;

	// Validate parameters.
	if ( report == NULL ) {
		return exception_throw("report was NULL.", function_name);
	}

	// Instantiate each section of the report.
	// Serpent section.
	exception = section_init(&(report->sections[REPORT_SECTION_SERPENT]), SERPENT);
	if ( exception != NULL ) {
		return exception_append(exception, function_name);
	}

	// Return success.
	return NULL;
}


exception_t* report_free(report_t* report) {
	char* function_name = "report_free()";
	exception_t* exception;
	int i;

	// Validate parameters.
	if ( report == NULL ) {
		return exception_throw("report was NULL.", function_name);
	}

	// Free each section of the report.
	for ( i = 0; i < REPORT_SECTION_COUNT; i++ ) {
		exception = section_free(&(report->sections[i]));
		if ( exception != NULL ) {
			return exception_append(exception, function_name);
		}
	}

	// Return success.
	return NULL;
}


exception_t* report_write(report_t* report, char* filepath) { 
	char* function_name = "report_write()";

	return exception_throw("Not implemented.", function_name);
}

