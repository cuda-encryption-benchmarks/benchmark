#include "algorithm.h"


exception_t* algorithm_get_name(enum algorithm algorithm, char* name) {
	char* function_name = "algorithm_get_name()";

	// Validate parameters.
	if ( name == NULL ) {
		return exception_throw("name was NULL.", function_name);
	}

	// Get the name of the specified algorithm.
	switch(algorithm) {
	case AES:
		strcpy(name, "Advanced Encryption Standard (AES)");
		break;
	case SERPENT:
		strcpy(name, "Serpent");
		break;
	case TWOFISH:
		strcpy(name, "Twofish");
		break;
	default:
		return exception_throw("Unsupported algorithm.", function_name);
	}

	// Return success.
	return NULL;
}

