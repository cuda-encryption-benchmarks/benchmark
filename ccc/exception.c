/*
 * Original Author: Wade Cline (clinew)
 * File: exception_t.c
 * Created: 2011 October 5 by clinew
 * Last Modified: 2012 February 19, by clinew
 * 
 * See: "exception.h".
 */	


#include "exception.h"


/**	Internal string append function.
 * 	@return Pointer to a newly allocated character array containing the concatenation of the head and tail.
 */
char* exception_string_append(char* head, char* tail) {
	char* appended_string = NULL;
	int head_length = -1;
	int tail_length = -1;

	// Check the neither of the char arrays are null.
	if ( head == NULL || tail == NULL ) {
		fprintf(stderr, "exception_string_append failed: Destination or source string was null.\n");
		return NULL;
	}

	// Get the length of each string.
	head_length = strlen(head);
	tail_length = strlen(tail);

	// Allocate space for the two appended strings. Equal to the length of both strings combined, plus one for the null character.
	appended_string = (char*)malloc((sizeof(char)*(head_length + tail_length))+1);
	if ( appended_string == NULL ) {
		fprintf(stderr, "exception_string_append failed: Unable to allocate space for the new string.\n");
		return NULL;
	}

	// Copy the head char array to the newly allocated char array.
	strcpy(appended_string, head);

	// Concatenate the tail char array to the newly allocated char array.
	strcat(appended_string, tail);

	// Return the concatenated char array.
	return appended_string;
}


/**	Internal string copy function.
 * 	@return	Pointer to a newly-allocated character array.
 */
char* exception_string_copy(char* source) {
	char* copied_string = NULL;
	int length = -1;

	// Check that the source character array is not null.
	if ( source == NULL ) {
		fprintf(stderr, "exception_string_copy failed: specified string was NULL.\n");
		return NULL;
	}

	// Get the length of the string.
	length = strlen(source);

	// Allocate memory for the new char array.
	copied_string = (char*)malloc((sizeof(char)*length)+1);
	if ( copied_string == NULL ) {
		fprintf(stderr, "exception_string_copy failed: Unable to allocate memory.\n");
		return NULL;
	}

	// Copy the source char array into the newly allocated char array.
	strcpy( copied_string, source );

	// Return the newly allocated char array.
	return copied_string;
}


exception_t* exception_append(exception_t* exception, char* function_name) {
	char* temp_string = NULL;

	// Check that exception is non-NULL.
	if ( exception == NULL ) {
		fprintf(stderr, "exception_append failed: exception is NULL.\n");
		fflush(stderr);
		exit(-1);
	}

	// Check that function_name is non-NULL.
	if ( function_name == NULL ) {
		fprintf(stderr, "exception_append failed: function_name is NULL.\n");
		fflush(stderr);
		exit(-1);
	}

	// Add a newline, tab, and list marker to exception_t's stack_trace.
	temp_string = exception->stack_trace;
	exception->stack_trace = exception_string_append(temp_string, "\n\t-");
	if ( exception->stack_trace == NULL ) {
		fprintf(stderr, "exception_append failed: unable to append " \
			"\"\\n\".\n");
		fflush(stderr);
		exit(-1);
	}
	free(temp_string);
	
	// Add function_name to exception_t's stack_trace.
	temp_string = exception->stack_trace;
	exception->stack_trace = exception_string_append(temp_string, function_name);
	if ( exception->stack_trace == NULL ) {
		fprintf(stderr, "exception_append failed: unable to append " \
			"function_name.\n");
		fflush(stderr);
		exit(-1);
	}
	free(temp_string);

	// Return the exception_t object.
	return exception;
}


exception_t* exception_catch(exception_t* exception) {
	// Check that the exception is non-NULL.
	if ( exception == NULL ) {
		fprintf(stderr, "exception_catch failed: exception_t is NULL.\n");
		fflush(stderr);
		exit(-1);
	}

	// Print the exception message.
	exception_print(exception);

	// Free the exception and return NULL (see: exception_free()).
	return exception_free(exception);
}


exception_t* exception_free(exception_t* exception) {
	// Check that the exception is non-NULL.
	if ( exception == NULL ) {
		fprintf(stderr, "exception_free failed: exception_t is NULL.\n");
		fflush(stderr);
		exit(-1);
	}

	// Free the specified exception_t's members.
	free(exception->message);
	free(exception->stack_trace);

	// Free the exception itself.
	free(exception);

	// Return success.
	return NULL;
}


exception_t* exception_print(exception_t* exception) {
	// Check that the exception is non-NULL.
	if ( exception == NULL ) {
		fprintf(stderr, "exception_print failed: exception_t is NULL.\n");
		fflush(stderr);
		exit(-1);
	}

	// Print the exception to stdout.
	fprintf(stdout, "\n*** EXCEPTION ***\n" \
		"Message: %s\nStack Trace: \n\t-%s\n" \
		"*****************\n\n", \
		exception->message, exception->stack_trace);

	// Return success.
	return NULL;
}


exception_t* exception_throw(char* message, char* function_name) {
	// Check that message is non-NULL.
	if ( message == NULL ) {
		fprintf(stderr, "exception_throw failed: message parameter " \
			"was NULL.\n");
		fflush(stderr);
		exit(-1);
	}

	// Check that function_name is non-NULL.
	if ( function_name == NULL ) {
		fprintf(stderr, "exception_throw failed: function_name parameter " \
			"was NULL.\n");
		fflush(stderr);
		exit(-1);
	}

	// Malloc space for the new exception.
	exception_t* exception = (exception_t*)malloc(sizeof(exception_t));
	if ( exception == NULL ) {
		fprintf(stderr, "Malloc failed during exception creation.");
		fflush(stderr);
		exit(-1);
	}

	// Assign exception_t's message.
	exception->message = exception_string_copy(message);
	if ( exception->message == NULL ) {
		fprintf(stderr, "exception_throw failed: exception_string_copy returned NULL " \
			 "when copying message.\n");
		exit(-1);
	}

	// Assign exception_t's function_name.
	exception->stack_trace = exception_string_copy(function_name);
	if ( exception->message == NULL ) {
		fprintf(stderr, "exception_throw failed: exception_string_copy returned NULL " \
			"when copying function_name.\n");
	}

	// Return exception_t.
	return exception;
}

