/*
 * Original Author: Wade Cline (clinew)
 * File: exception.h
 * Created: 2011 October 5 by clinew
 * Last Modified: 2012 February 19, by clinew
 *
 * This file contains functions related to the exception structure. The 
 * exception structure implements ad hoc exception handling in C. This 
 * allows the easy passing of messages and a stack trace to the calling 
 * function for easier debugging.
 */


#ifndef exception_H
#define exception_H


#include <string.h>
#include <stdio.h>
#include <stdlib.h>


/**	Abstract for an exception class. Contains a message and stack trace.
 */
typedef struct {
	// Contains the programmer's message for the exception.
	char* message;
	// Contains the stack trace of the exception.
	char* stack_trace;
} exception_t;


/**	Appends the specified function name to the exception_t's stack_trace
 *	and returns the exception_t. Will exit the program on error.
 *	@out	exception has the specified function name appended 
 *		to its stack trace.
 *	@return	Returns the specified exception.
 */
exception_t* exception_append(exception_t* exception, char* function_name);


/**	Print the exception message and stack trace to standard out, and frees
 *	the exception. Calls exception_print and exception_free. Returns NULL.
 *	@out	exception is freed.
 *	@return	The return value of exception_free().
 */
exception_t* exception_catch(exception_t* exception);


/**	Free the specified exception and its members.
 *	@return	NULL. This is done so that the pointer need not be set to NULL
 *	on the following line.
 */
exception_t* exception_free(exception_t* exception);


/**	Prints the specified exception to stdout.
 */
exception_t* exception_print(exception_t* exception);


/**	Mallocs and returns a pointer to a new exception_t type.
 *	@param	message: A special message for the exception.
 *	@param	function_name: The name of the function in which the exception_t
 *		was thrown.
 */
exception_t* exception_throw(char* message, char* function_name);


#endif
