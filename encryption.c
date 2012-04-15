#include "encryption.h"


exception_t* encryption_get_name(enum encryption encryption, char* name) {
	char* function_name = "encryption_get_name()";

	// Validate parameters.
	if ( name == NULL ) {
		return exception_throw("name was NULL.", function_name);
	}

	// Copy the corresponding name
	switch(encryption) {
	case DECRYPT:
		strcpy(name, "Decryption");
		break;
	case ENCRYPT:
		strcpy(name, "Encryption");
		break;
	default:
		return exception_throw("Unsupported parameter.", function_name);
	}

	// Return success.
	return NULL;
}

