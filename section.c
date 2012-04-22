#include "section.h"


exception_t* section_execute(section_t* section, char* input_filepath) {
	char* function_name = "section_execute()";
	exception_t* exception;
	char algorithm_name[ALGORITHM_NAME_LENGTH];
	int i;

	// Validate parameters.
	if ( section == NULL ) {
		return exception_throw("section was NULL.", function_name);
	}
	if ( input_filepath == NULL ) {
		return exception_throw("input_filepath was NULL.", function_name);
	}

	// Get algorithm name.
	exception = algorithm_get_name(section->algorithm, algorithm_name);
	if ( exception != NULL ) {
		return exception_append(exception, function_name);
	}

	// Execute each section.
	fprintf(stdout, "  Section \"%s\" execution starting.\n", algorithm_name);
	for ( i = 0; i < SECTION_SUBSECTION_COUNT; i++ ) {
		exception = subsection_execute(&(section->subsections[i]), &(section->key), input_filepath, section->algorithm);
		if ( exception != NULL ) {
			return exception_append(exception, function_name);
		}
	}

	// Return success.
	fprintf(stdout, "  Section \"%s\" execution completed successfully.\n", algorithm_name);
	return NULL;
}


exception_t* section_init(section_t* section, int data_count, enum algorithm algorithm) {
	char* function_name = "section_init()";
	exception_t* exception;

	// Validate parameters.
	if ( section == NULL ) {
		return exception_throw("section was NULL.", function_name);
	}
	if ( data_count < 1 ) {
		return exception_throw("iteration_count must be > 0.", function_name);
	}

	// Set the algorithm attribute.
	section->algorithm = algorithm;

	// Initialize the key.
	section->key.key0.x0 = 0xdeadbeef;
	section->key.key0.x1 = 0xdeadbeef;
	section->key.key0.x2 = 0xdeadbeef;
	section->key.key0.x3 = 0xdeadbeef;
	section->key.key1.x0 = 0xdeadbeef;
	section->key.key1.x1 = 0xdeadbeef;
	section->key.key1.x2 = 0xdeadbeef;
	section->key.key1.x3 = 0xdeadbeef;

	// Initialize the subsections.
	// Serial.
	exception = subsection_init(&(section->subsections[SECTION_SUBSECTION_SERIAL]), data_count, SERIAL);
	if ( exception != NULL ) {
		return exception_append(exception, function_name);
	}
	// Parallel.
	exception = subsection_init(&(section->subsections[SECTION_SUBSECTION_PARALLEL]), data_count, PARALLEL);
	if ( exception != NULL ) {
		return exception_append(exception, function_name);
	}
	// CUDA.
	exception = subsection_init(&(section->subsections[SECTION_SUBSECTION_CUDA]), data_count, CUDA);
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

	// Set key to predetermined value.
	section->key.key0.x0 = 0;
	section->key.key0.x1 = 0;
	section->key.key0.x2 = 0;
	section->key.key0.x3 = 0;
	section->key.key1.x0 = 0;
	section->key.key1.x1 = 0;
	section->key.key1.x2 = 0;
	section->key.key1.x3 = 0;

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
	char algorithm_name[50];
	exception_t* exception;
	int i;

	// Validate parameters.
	if ( section == NULL ) {
		return exception_throw("section was NULL.", function_name);
	}
	if ( file == NULL ) {
		return exception_throw("file was NULL.", function_name);
	}

	// Get algorithm name.
	exception = algorithm_get_name(section->algorithm, algorithm_name);
	if ( exception != NULL ) {
		return exception_append(exception, function_name);
	}
	fprintf(stdout, "  Section \"%s\" writing starting.\n", algorithm_name);

	// Write section head.
	fprintf(file, "\\subsection{%s}\n", algorithm_name);

	// Write subsection data.
	for ( i = 0; i < SECTION_SUBSECTION_COUNT; i++ ) {
		exception = subsection_write(&(section->subsections[i]), file, section->algorithm);
		if ( exception != NULL ) {
			return exception_append(exception, function_name);
		}
	}

	// Return success.
	fprintf(stdout, "  Section \"%s\" written.\n", algorithm_name);
	return NULL;
}

