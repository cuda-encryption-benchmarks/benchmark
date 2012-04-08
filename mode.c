#include "mode.h"


exception_t* mode_get_name(enum mode mode, char* name) {
	char* function_name = "mode_get_name()";

	// Validate parameters.
	if ( name == NULL ) {
		return exception_throw("name was NULL.", function_name);
	}

	// Get the mode name.
	switch(mode) {
	case CUDA:
		strcpy(name, "Compute Unified Device Architecture (CUDA)");
		break;
	case PARALLEL:
		strcpy(name, "Parallel");
		break;
	case SERIAL:
		strcpy(name, "Serial");
		break;
	default:
		return exception_throw("Unsupported mode.", function_name);
	}

	// Return success.
	return NULL;
}

