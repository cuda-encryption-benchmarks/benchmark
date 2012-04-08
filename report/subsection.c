
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
	char mode_name[50];
	exception_t* exception;
	int i;

	// Validate parameters.
	if ( subsection == NULL ) {
		return exception_throw("subsection was NULL.", function_name);
	}
	if ( file == NULL ) {
		return exception_throw("file was NULL.", function_name);
	}

	// Get the mode of the subsection.
	exception = mode_get_name(subsection->mode, mode_name);
	if ( exception != NULL ) {
		return exception_append(exception, function_name);
	}

	// Write subsection head.
	fprintf(file, "\\subsubsection{%s}\n", mode_name);

	// Write a table of the subsection data.
	fprintf(file, "\\begin{tabular}[c]{r|r|r|r}\n");
	fprintf(file, "Iteration \\# & Elapsed Seconds & Elapsed Nanoseconds & Global Memory Used \\\\\n\\hline\n");
	for ( i = 0; i < SUBSECTION_ITERATION_COUNT; i++ ) {
		fprintf(file, "%i & %li & %li & %lli \\\\\n", i + 1, subsection->data_encrypt[i].time_elapsed.tv_sec, subsection->data_encrypt[i].time_elapsed.tv_nsec, subsection->data_encrypt[i].buffer_size);
	}
	fprintf(file, "\\hline\n");
	fprintf(file, "\\end{tabular}\n");

	// Return success.
	return NULL;
}

