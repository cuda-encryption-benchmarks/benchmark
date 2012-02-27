#include "serpent.h"


void sbox0_encrypt(uint32* r0, uint32* r1, uint32* r2, uint32* r3, uint32* r4) {
        *r3 ^= *r0;
        *r4 = *r1;
        *r1 &= *r3;
        *r4 ^= *r2;
        *r1 ^= *r0;
        *r0 |= *r3;
        *r0 ^= *r4;
        *r4 ^= *r3;
        *r3 ^= *r2;
        *r2 |= *r1;
        *r2 ^= *r4;
        *r4 = ~*r4;
        *r4 |= *r1;
        *r1 ^= *r3;
        *r1 ^= *r4;
        *r3 |= *r0;
        *r1 ^= *r3;
        *r4 ^= *r3;
}


void sbox1_encrypt(uint32* r0, uint32* r1, uint32* r2, uint32* r3, uint32* r4) {
    *r0 = ~*r0;
    *r2 = ~*r2;
    *r4 = *r0;
    *r0 &= *r1;
    *r2 ^= *r0;
    *r0 |= *r3;
    *r3 ^= *r2;
    *r1 ^= *r0;
    *r0 ^= *r4;
    *r4 |= *r1;
    *r1 ^= *r3;
    *r2 |= *r0;
    *r2 &= *r4;
    *r0 ^= *r1;
    *r1 &= *r2;
    *r1 ^= *r0;
    *r0 &= *r2;
    *r0 ^= *r4;
}


void sbox2_encrypt(uint32* r0, uint32* r1, uint32* r2, uint32* r3, uint32* r4) {
        *r4 = *r0;
        *r0 &= *r2;
        *r0 ^= *r3;
        *r2 ^= *r1;
        *r2 ^= *r0;
        *r3 |= *r4;
        *r3 ^= *r1;
        *r4 ^= *r2;
        *r1 = *r3;
        *r3 |= *r4;
        *r3 ^= *r0;
        *r0 &= *r1;
        *r4 ^= *r0;
        *r1 ^= *r3;
        *r1 ^= *r4;
        *r4 = ~*r4;
}


void sbox3_encrypt(uint32* r0, uint32* r1, uint32* r2, uint32* r3, uint32* r4) {
        *r4 = *r0;
        *r0 |= *r3;
        *r3 ^= *r1;
        *r1 &= *r4;
        *r4 ^= *r2;
        *r2 ^= *r3;
        *r3 &= *r0;
        *r4 |= *r1;
        *r3 ^= *r4;
        *r0 ^= *r1;
        *r4 &= *r0;
        *r1 ^= *r3;
        *r4 ^= *r2;
        *r1 |= *r0;
        *r1 ^= *r2;
        *r0 ^= *r3;
        *r2 = *r1;
        *r1 |= *r3;
        *r1 ^= *r0;
}


void sbox4_encrypt(uint32* r0, uint32* r1, uint32* r2, uint32* r3, uint32* r4) {
        *r1 ^= *r3;
        *r3 = ~*r3;
        *r2 ^= *r3;
        *r3 ^= *r0;
        *r4 = *r1;
        *r1 &= *r3;
        *r1 ^= *r2;
        *r4 ^= *r3;
        *r0 ^= *r4;
        *r2 &= *r4;
        *r2 ^= *r0;
        *r0 &= *r1;
        *r3 ^= *r0;
        *r4 |= *r1;
        *r4 ^= *r0;
        *r0 |= *r3;
        *r0 ^= *r2;
        *r2 &= *r3;
        *r0 = ~*r0;
        *r4 ^= *r2;
}


void sbox5_encrypt(uint32* r0, uint32* r1, uint32* r2, uint32* r3, uint32* r4) {
        *r0 ^= *r1;
        *r1 ^= *r3;
        *r3 = ~*r3;
        *r4 = *r1;
        *r1 &= *r0;
        *r2 ^= *r3;
        *r1 ^= *r2;
        *r2 |= *r4;
        *r4 ^= *r3;
        *r3 &= *r1;
        *r3 ^= *r0;
        *r4 ^= *r1;
        *r4 ^= *r2;
        *r2 ^= *r0;
        *r0 &= *r3;
        *r2 = ~*r2;
        *r0 ^= *r4;
        *r4 |= *r3;
        *r2 ^= *r4;
}


void sbox6_encrypt(uint32* r0, uint32* r1, uint32* r2, uint32* r3, uint32* r4) {
        *r2 = ~*r2;
        *r4 = *r3;
        *r3 &= *r0;
        *r0 ^= *r4;
        *r3 ^= *r2;
        *r2 |= *r4;
        *r1 ^= *r3;
        *r2 ^= *r0;
        *r0 |= *r1;
        *r2 ^= *r1;
        *r4 ^= *r0;
        *r0 |= *r3;
        *r0 ^= *r2;
        *r4 ^= *r3;
        *r4 ^= *r0;
        *r3 = ~*r3;
        *r2 &= *r4;
        *r2 ^= *r3;
}


void sbox7_encrypt(uint32* r0, uint32* r1, uint32* r2, uint32* r3, uint32* r4) {
        *r4 = *r2;
        *r2 &= *r1;
        *r2 ^= *r3;
        *r3 &= *r1;
        *r4 ^= *r2;
        *r2 ^= *r1;
        *r1 ^= *r0;
        *r0 |= *r4;
        *r0 ^= *r2;
        *r3 ^= *r1;
        *r2 ^= *r3;
        *r3 &= *r0;
        *r3 ^= *r4;
        *r4 ^= *r2;
        *r2 &= *r0;
        *r4 = ~*r4;
        *r2 ^= *r4;
        *r4 &= *r0;
        *r1 ^= *r3;
        *r4 ^= *r1;
}


exception_t* serpent_encrypt_serial(block128* user_key, block128* blocks, int block_count) {
	char* function_name = "serpent_serial()";
	exception_t* exception;
	block128* serpent_key;

	// Generate the key.
	exception = serpent_generate_key_serial(user_key, &serpent_key);
	if ( exception != NULL ) {
		return exception_append(exception, function_name);
	}

	// Encrypt each block.
	for ( int i = 0; i < block_count; i++ ) {
		// Apply first 31 S-boxes.
			// Key mixing.
			// S-boxes.
			// Linear transformation.

		// Apply final S-box.
	}

	return exception_throw("Function not implemented.", function_name);
}


exception_t* serpent_generate_key_serial(block128* user_key, block128** serpent_key) {
	char* function_name = "serpent_generate_key_serial()";

	return exception_throw("Function not implemented.", function_name);
}


