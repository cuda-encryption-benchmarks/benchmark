
//! Macro to allow clock_getres() and related functionality.
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 199309L
#endif
//! Open large files.
#define _LARGEFILE64_SOURCE
//! For lstat64().
#define _BSD_SOURCE


#include <errno.h>
#include <fcntl.h>
#include <time.h>
#include <unistd.h>

#include "aes.h"
#include "block128.h"
#include "ccc/ccc.h"
#include "file.h"
#include "mirror_bytes.h"
#include "serpent.h"
#include "report.h"
#include "typedef.h"


exception_t* arguments_validate(int argc, char* argv[]) {
	//char* function_name = "arguments_validate()";

	// Validate argument count.
	if ( argc != 3 ) {
		print_usage("Invalid number of arguments.");
	}

	// Return success.
	return NULL;
}


exception_t* arguments_parse(char* argv[], int* data_count) {
	char* function_name = "arguments_parse()";
	exception_t* exception;

	// Validate parameters.
	if ( argv == NULL ) {
		return exception_throw("argv was NULL.", function_name);
	}
	if ( data_count == NULL ) {
		return exception_throw("data_count was NULL.", function_name);
	}
	
	// Parse the data count.
	exception = arguments_parse_data_count(argv[1], data_count);
	if ( exception != NULL ) {
		return exception_append(exception, function_name);
	}

	// Check that the file exists
	exception = arguments_parse_input_filepath(argv[2]);
	if ( exception != NULL ) {
		return exception_append(exception, function_name);
	}

	// Return success.
	return NULL;
}


exception_t* arguments_parse_data_count(char* argument, int* data_count) {
	char* function_name = "arguments_parse_data_count()";
	int temp;

	// Validate parameters.
	if ( argument == NULL ) {
		return exception_throw("argument was NULL.", function_name);
	}
	if ( data_count == NULL ) {
		return exception_throw("data_count was NULL.", function_name);
	}

	// Parse the argument.
	temp = atoi(argument);
	if ( temp < 1 ) {
		print_usage("The number of iterations must be greater than zero.");
	}

	// Assign output parameter.
	(*data_count) = temp;

	// Return success.
	return NULL;
}


exception_t* arguments_parse_input_filepath(char* argument) {
	char* function_name = "arguments_parse_input_filepath()";
	struct stat stats;

	// Validate parameters.
	if ( argument == NULL ) {
		return exception_throw("argument was NULL.", function_name);
	}

	// Check that the file exists.
        if ( stat(argument, &stats) == 0 ) { // File exists.
                // Check if the file is a directory.
                if ( S_ISDIR(stats.st_mode) ) {
                        return exception_throw("The specified file is a directory.", function_name);
                }
        }
        else { // Stat failed.
                if ( errno == ENOENT ) { // File does not exist.
                        return exception_throw("The specified file does not exist.", function_name);
                }
                else { // Any other error.
                        perror(NULL);
                        return exception_throw("Unexpected error when getting file stats.", function_name);
                }
        }

	// Return success.
	return NULL;
}


void print_usage(char* message) {
	// Print usage message.
	fprintf(stdout, "\a\nERROR: %s\n\n", message);
	fprintf(stdout, "--- CUDA Benchmarks Report ---\n");
	fprintf(stdout, "USAGE: ./benchmark <iterations> <input>\n");
	fprintf(stdout, "\titerations: The number of iterations to run on the specified input.\n");
	fprintf(stdout, "\tinput: The input file.\n\n");

	// Exit the program.
	exit(EXIT_FAILURE);
}


int main( int argc, char* argv[] ) {
	exception_t* exception;
	report_t report;
	char basepath[REPORT_BASEPATH_LENGTH_MAX];
	int data_count;
	int status;

	// Validate arguments.
	arguments_validate(argc, argv);

	// Parse the arguments.
	exception = arguments_parse(argv, &data_count);
	if ( exception != NULL ) {
		print_usage(exception->message);
	}

	// Initialize the report.
	fprintf(stdout, "Beginning automatic report generation.\n");
	status = EXIT_SUCCESS;
	exception = report_init(&report, argv[2], data_count);
	if ( exception != NULL ) {
		exception_catch(exception);
		fprintf(stdout, "ERROR in report initialization (see above).\n");
		exit(EXIT_FAILURE);
	}
	strcpy(basepath, report.basepath);

	// Execute the report.
	exception = report_execute(&report);
	if ( exception != NULL ) {
		exception_catch(exception);
		status = EXIT_FAILURE;
		fprintf(stdout, "ERROR in report execution (see above). Attempting write...\n");
	}

	// Write the report.
	exception = report_write(&report);
	if ( exception != NULL ) {
		exception_catch(exception);
		status = EXIT_FAILURE;
		fprintf(stdout, "ERROR writing the report.\n");
	}

	// Free the report.
	exception = report_free(&report);
	if ( exception != NULL ) {
		exception_catch(exception);
		status = EXIT_FAILURE;
		fprintf(stdout, "ERROR freeing the report; luckily, it's probably not important...\n");
	}

	// Gracefully exit the program.
	fprintf(stdout, "Finished automatic report generation.\n" \
		"Results may be viewed in the \"%s\" directory.\n", basepath);
	exit(status);
}
