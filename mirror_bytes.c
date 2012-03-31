#include "mirror_bytes.h"


uint32_t mirror_bytes32(uint32_t x) { 
	uint32_t n;

	// Flip the bytes.
	n = (uint8_t) x;
        n <<= 8; n |= (uint8_t) (x >> 8);
        n <<= 8; n |= (uint8_t) (x >> 16);
        n = (n << 8) | (uint8_t) (x >> 24);

	// Return the mirrored bytes.
	return n;
}

