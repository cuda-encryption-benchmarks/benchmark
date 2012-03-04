#include "mirror_bytes.h"


exception_t* mirror_bytes32(uint32 x, uint32* out) { 
	char* function_name = "mirror_bytes32()";
	uint32 n;
	
	// Validate parameters.
	if ( out == NULL ) {
		return exception_throw("out was NULL.", function_name);
	}

	// Flip the bytes.
	n = (uint8_t) x;
        n <<= 8; n |= (uint8_t) (x >> 8);
        n <<= 8; n |= (uint8_t) (x >> 16);
        n = (n << 8) | (uint8_t) (x >> 24);

	// Assign output parameter.
	(*out) = n;

	// Return success.
	return NULL;
}

