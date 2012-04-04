
#include "subsection.h"


exception_t* subsection_init(subsection_t* subsection, enum mode mode) {
	char* function_name = "subsection_init()";
	int i;

	// Validate parameters.
	if ( subsection == NULL ) {
		return exception_throw("subsection was NULL.", function_name);
	}

	// Set the mode.
	subsection->mode = mode;

	// Initialize the data to have impossible values.
	for ( i = 0; i < SUBSECTION_ITERATION_COUNT; i++ ) {
		subsection->data_encrypt[i].time_elapsed.tv_sec = -1;
		subsection->data_encrypt[i].time_elapsed.tv_nsec = -1;
		subsection->data_encrypt[i].buffer_size = -1;
		subsection->data_decrypt[i].time_elapsed.tv_sec = -1;
		subsection->data_decrypt[i].time_elapsed.tv_nsec = -1;
		subsection->data_decrypt[i].buffer_size = -1;
	}

	// Return success.
	return NULL;
}


exception_t* subsection_free(subsection_t* subsection) {
	char* function_name = "subsection_free()";
	int i;

	// Validate parameters.
	if ( subsection == NULL ) {
		return exception_throw("subsection was NULL.", function_name);
	}

	// Set mode to an impossible value.
	subsection->mode = -1;

	// Set the data to impossible values.
	for ( i = 0; i < SUBSECTION_ITERATION_COUNT; i++ ) {
		subsection->data_encrypt[i].time_elapsed.tv_sec = -1;
		subsection->data_encrypt[i].time_elapsed.tv_nsec = -1;
		subsection->data_encrypt[i].buffer_size = -1;
		subsection->data_decrypt[i].time_elapsed.tv_sec = -1;
		subsection->data_decrypt[i].time_elapsed.tv_nsec = -1;
		subsection->data_decrypt[i].buffer_size = -1;
	}

	// Return success.
	return NULL;
}


exception_t* subsection_write(subsection_t* subsection, FILE* file) {
	char* function_name = "subsection_write()";

	return exception_throw("Not implemented.", function_name);
}

