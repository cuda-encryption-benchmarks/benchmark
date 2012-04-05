#include "report.h"


exception_t* report_init(report_t* report) {
	char* function_name = "report_init()";
	char filepath[REPORT_BASEPATH_LENGTH_MAX + REPORT_FILENAME_LENGTH_MAX + 3];
	exception_t* exception;

	// Validate parameters.
	if ( report == NULL ) {
		return exception_throw("report was NULL.", function_name);
	}

	// Set the report name.
	strcpy(report->filename, "report");

	// Create the directory structure.
	report->basepath[0] = '\0';
	exception = report_init_create_directories(report);
	if ( exception != NULL ) {
		return exception_append(exception, function_name);
	}
	
	// Open the report file.
	strcat(filepath, report->basepath);
	strcat(filepath, report->filename);
	strcat(filepath, ".tex");
	report->file = fopen(filepath, "w");
	if ( report->file == NULL ) {
		return exception_throw("Unable to create report file.", function_name);
	}

	// Initialize each section of the report.
	// Serpent section.
	exception = section_init(&(report->sections[REPORT_SECTION_SERPENT]), SERPENT);
	if ( exception != NULL ) {
		return exception_append(exception, function_name);
	}

	// Return success.
	return NULL;
}


exception_t* report_init_create_directories(report_t* report) {
	char* function_name = "report_init_create_directories()";
	struct stat stats;
	time_t time_raw;
	struct tm time_current;
	char time_buffer[50];

	// Validate parameters.
	if ( report == NULL ) {
		return exception_throw("report was NULL.", function_name);
	}

	// Make/ensure creation of results directory.
	strcpy(report->basepath, "results");
	if ( stat(report->basepath, &stats) == 0 ) { // File exists.
		// Check if the file is a directory.
		if ( !(S_ISDIR(stats.st_mode)) ) {
			return exception_throw("The results file exists and is not a directory.", function_name);
		}
	}
	else { // Stat failed.
		if ( errno == ENOENT ) { // File does not exist.
			// Create the directory.
			if ( mkdir(report->basepath, 0700) == -1 ) {
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
	strcat(report->basepath, "/");
	strcat(report->basepath, time_buffer);
	strcat(report->basepath, "/");

	// Create the report directory.
	if ( mkdir(report->basepath, 0700) == -1 ) {
		perror(NULL);
		return exception_throw("Unable to create report directory.", function_name);
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

	// Invalidate the basepath.
	report->basepath[0] = '\0';

	// Invalidate the filename.
	report->filename[0] = '\0';

	// Free each section of the report.
	for ( i = 0; i < REPORT_SECTION_COUNT; i++ ) {
		exception = section_free(&(report->sections[i]));
		if ( exception != NULL ) {
			return exception_append(exception, function_name);
		}
	}

	// Close the report file.
	if ( fclose(report->file) == EOF ) {
		return exception_throw("Unable to close report file.", function_name);
	}	

	// Return success.
	return NULL;
}


exception_t* report_write(report_t* report) { 
	char* function_name = "report_write()";

	// Validate parameters.
	if ( report == NULL ) {
		return exception_throw("report was NULL.", function_name);
	}

	// Write output. DEBUG
	fprintf(report->file, "\\documentclass{article}\n\n");
	fprintf(report->file, "\\usepackage{amsmath}\n\n");
	fprintf(report->file, "\\title{CUDA Benchmarking Report}\n\\author{Automatically Generated}\n\n");
	fprintf(report->file, "\\begin{document}\n\n");
	fprintf(report->file, "\\maketitle\n\n");
	fprintf(report->file, "\\end{document}\n");

	// Return not implemented.
	return exception_throw("Not implemented.", function_name);
}


exception_t* report_write_compile_latex(report_t* report) {
	char* function_name = "report_write_compile_latex()";

	return exception_throw("Not implemented.", function_name);
}



