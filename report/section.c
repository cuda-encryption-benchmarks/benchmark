#include "section.h"


exception_t* section_init(section_t* section, enum algorithm algorithm) {
	char* function_name = "section_init()";
	exception_t* exception;

	// Validate parameters.
	if ( section == NULL ) {
		return exception_throw("section was NULL.", function_name);
	}

	// Set the algorithm attribute.
	section->algorithm = algorithm;

	// Initialize the subsections.
	// Serial.
	exception = subsection_init(&(section->subsections[SECTION_SUBSECTION_SERIAL]), SERIAL);
	if ( exception != NULL ) {
		return exception_append(exception, function_name);
	}
	// Parallel.
	exception = subsection_init(&(section->subsections[SECTION_SUBSECTION_SERIAL]), PARALLEL);
	if ( exception != NULL ) {
		return exception_append(exception, function_name);
	}
	// CUDA.
	exception = subsection_init(&(section->subsections[SECTION_SUBSECTION_CUDA]), CUDA);
	if ( exception != NULL ) {
		return exception_append(exception, function_name);
	}

	// Return success.
	return NULL;
}


exception_t* section_free(section_t* section) { 
	char* function_name = "section_free()";
	exception_t* exception;
	int i;

	// Validate parameters.
	if ( section == NULL ) {
		return exception_throw("section was NULL.", function_name);
	}

	// Set algorithm to an impossible value.
	section->algorithm = -1;

	// Free each subsection.
	for ( i = 0; i < SECTION_SUBSECTION_COUNT; i++ ) {
		exception = subsection_free(&(section->subsections[i]));
		if ( exception != NULL ) {
			return exception_append(exception, function_name);
		}
	}

	// Return success.
	return NULL;
}


exception_t* section_write(section_t* section, FILE* file) {
	char* function_name = "section_write()";

	return exception_throw("Not implemented.", function_name);
}

