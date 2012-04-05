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


exception_t* report_write(report_t* report) { 
	char* function_name = "report_write()";
	exception_t* exception;
	char report_filepath[50];

	// Validate parameters.
	if ( report == NULL ) {
		return exception_throw("report was NULL.", function_name);
	}

	// Create and get the report directory path.
	exception = report_write_create_directories(report, report_filepath);
	if ( exception != NULL ) {
		return exception_append(exception, function_name);
	}

	// Return not implemented.
	return exception_throw("Not implemented.", function_name);
}


exception_t* report_write_create_directories(report_t* report, char* report_filepath) {
	char* function_name = "report_write_create_directories()";
	struct stat stats;
	time_t time_raw;
	struct tm time_current;
	char time_buffer[50];

	// Validate parameters.
	if ( report == NULL ) {
		return exception_throw("report was NULL.", function_name);
	}
	if ( report_filepath == NULL ) {
		return exception_throw("report_filepath was NULL.", function_name);
	}

	// Make/ensure creation of results directory.
	strcpy(report_filepath, "results");
	if ( stat(report_filepath, &stats) == 0 ) { // File exists.
		// Check if the file is a directory.
		if ( !(S_ISDIR(stats.st_mode)) ) {
			return exception_throw("The results file exists and is not a directory.", function_name);
		}
	}
	else { // Stat failed.
		if ( errno == ENOENT ) { // File does not exist.
			// Create the directory.
			if ( mkdir(report_filepath, 0700) == -1 ) {
				perror(NULL);
				return exception_throw("Unable to create results directory.", function_name);
			}
		}
		else { // Any other error.
			// Return failure.
			perror(NULL);
			return exception_throw("Unable to check for results directory.", function_name);
		}
	}

	// Get the current time.
	if ( time(&time_raw) == -1 ) {
		return exception_throw("Unable to get current time.", function_name);
	}

	// Get the struct tm representation of the current time.
	if ( localtime_r(&time_raw, &time_current) == NULL ) {
		return exception_throw("Unable to get struct tm.", function_name);
	}

	// Get the string representation of the current time.
	if ( strftime(time_buffer, 50, "%Y_%m_%d_%H:%M:%S", &time_current) == 0 ) {
		return exception_throw("Unable to get string representation of current time.", function_name);
	}

	// Copy the string representation of the current time onto the report filepath.
	strcat(report_filepath, "/");
	strcat(report_filepath, time_buffer);
	strcat(report_filepath, "/");

	// Create the report directory.
	if ( mkdir(report_filepath, 0700) == -1 ) {
		perror(NULL);
		return exception_throw("Unable to create report directory.", function_name);
	}

	// Return success.
	return NULL;
}
