#include "report.h"


exception_t* report_init(report_t* report) {
	char* function_name = "report_init()";

	// Validate parameters.
	if ( report == NULL ) {
		return exception_throw("report was NULL.", function_name);
	}

	return exception_throw("Not implemented.", function_name);
}


exception_t* report_free(report_t* report) {
	char* function_name = "report_free()";

	return exception_throw("Not implemented.", function_name);
}


exception_t* report_write(report_t* report, char* filepath) { 
	char* function_name = "report_write()";

	return exception_throw("Not implemented.", function_name);
}

