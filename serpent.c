#include "serpent.h"


// order of output from S-box functions
#define beforeS0(f) f(0,a,b,c,d,e)
#define afterS0(f) f(1,b,e,c,a,d)
#define afterS1(f) f(2,c,b,a,e,d)
#define afterS2(f) f(3,a,e,b,d,c)
#define afterS3(f) f(4,e,b,d,c,a)
#define afterS4(f) f(5,b,a,e,c,d)
#define afterS5(f) f(6,a,c,b,e,d)
#define afterS6(f) f(7,a,c,d,b,e)
#define afterS7(f) f(8,d,e,b,a,c)

// order of output from inverse S-box functions
#define beforeI7(f) f(8,a,b,c,d,e)
#define afterI7(f) f(7,d,a,b,e,c)
#define afterI6(f) f(6,a,b,c,e,d)
#define afterI5(f) f(5,b,d,e,c,a)
#define afterI4(f) f(4,b,c,e,a,d)
#define afterI3(f) f(3,a,b,e,c,d)
#define afterI2(f) f(2,b,d,e,c,a)
#define afterI1(f) f(1,a,b,c,e,d)
#define afterI0(f) f(0,a,d,b,e,c)

// The linear transformation.
#define linear_transformation(i,a,b,c,d,e) {\
	a = rotl_fixed(a, 13);   \
	c = rotl_fixed(c, 3);    \
	d = rotl_fixed(d ^ c ^ (a << 3), 7);     \
	b = rotl_fixed(b ^ a ^ c, 1);    \
	a = rotl_fixed(a ^ b ^ d, 5);	    \
	c = rotl_fixed(c ^ d ^ (b << 7), 22);}

// The inverse linear transformation.
#define inverse_linear_transformation(i,a,b,c,d,e)        {\
        c = rotr_fixed(c, 22);   \
        a = rotr_fixed(a, 5);    \
        c ^= d ^ (b << 7);      \
        a ^= b ^ d;             \
        b = rotr_fixed(b, 1);    \
        d = rotr_fixed(d, 7) ^ c ^ (a << 3);     \
        b ^= a ^ c;             \
        c = rotr_fixed(c, 3);    \
        a = rotr_fixed(a, 13);}


#define S0(i, r0, r1, r2, r3, r4) \
       {	   \
    r3 ^= r0;   \
    r4 = r1;   \
    r1 &= r3;   \
    r4 ^= r2;   \
    r1 ^= r0;   \
    r0 |= r3;   \
    r0 ^= r4;   \
    r4 ^= r3;   \
    r3 ^= r2;   \
    r2 |= r1;   \
    r2 ^= r4;   \
    r4 = ~r4;      \
    r4 |= r1;   \
    r1 ^= r3;   \
    r1 ^= r4;   \
    r3 |= r0;   \
    r1 ^= r3;   \
    r4 ^= r3;   \
	    }

#define I0(i, r0, r1, r2, r3, r4) \
       {	   \
    r2 = ~r2;      \
    r4 = r1;   \
    r1 |= r0;   \
    r4 = ~r4;      \
    r1 ^= r2;   \
    r2 |= r4;   \
    r1 ^= r3;   \
    r0 ^= r4;   \
    r2 ^= r0;   \
    r0 &= r3;   \
    r4 ^= r0;   \
    r0 |= r1;   \
    r0 ^= r2;   \
    r3 ^= r4;   \
    r2 ^= r1;   \
    r3 ^= r0;   \
    r3 ^= r1;   \
    r2 &= r3;   \
    r4 ^= r2;   \
	    }

#define S1(i, r0, r1, r2, r3, r4) \
       {	   \
    r0 = ~r0;      \
    r2 = ~r2;      \
    r4 = r0;   \
    r0 &= r1;   \
    r2 ^= r0;   \
    r0 |= r3;   \
    r3 ^= r2;   \
    r1 ^= r0;   \
    r0 ^= r4;   \
    r4 |= r1;   \
    r1 ^= r3;   \
    r2 |= r0;   \
    r2 &= r4;   \
    r0 ^= r1;   \
    r1 &= r2;   \
    r1 ^= r0;   \
    r0 &= r2;   \
    r0 ^= r4;   \
	    }

#define I1(i, r0, r1, r2, r3, r4) \
       {	   \
    r4 = r1;   \
    r1 ^= r3;   \
    r3 &= r1;   \
    r4 ^= r2;   \
    r3 ^= r0;   \
    r0 |= r1;   \
    r2 ^= r3;   \
    r0 ^= r4;   \
    r0 |= r2;   \
    r1 ^= r3;   \
    r0 ^= r1;   \
    r1 |= r3;   \
    r1 ^= r0;   \
    r4 = ~r4;      \
    r4 ^= r1;   \
    r1 |= r0;   \
    r1 ^= r0;   \
    r1 |= r4;   \
    r3 ^= r1;   \
	    }

#define S2(i, r0, r1, r2, r3, r4) \
       {	   \
    r4 = r0;   \
    r0 &= r2;   \
    r0 ^= r3;   \
    r2 ^= r1;   \
    r2 ^= r0;   \
    r3 |= r4;   \
    r3 ^= r1;   \
    r4 ^= r2;   \
    r1 = r3;   \
    r3 |= r4;   \
    r3 ^= r0;   \
    r0 &= r1;   \
    r4 ^= r0;   \
    r1 ^= r3;   \
    r1 ^= r4;   \
    r4 = ~r4;      \
	    }

#define I2(i, r0, r1, r2, r3, r4) \
       {	   \
    r2 ^= r3;   \
    r3 ^= r0;   \
    r4 = r3;   \
    r3 &= r2;   \
    r3 ^= r1;   \
    r1 |= r2;   \
    r1 ^= r4;   \
    r4 &= r3;   \
    r2 ^= r3;   \
    r4 &= r0;   \
    r4 ^= r2;   \
    r2 &= r1;   \
    r2 |= r0;   \
    r3 = ~r3;      \
    r2 ^= r3;   \
    r0 ^= r3;   \
    r0 &= r1;   \
    r3 ^= r4;   \
    r3 ^= r0;   \
	    }

#define S3(i, r0, r1, r2, r3, r4) \
       {	   \
    r4 = r0;   \
    r0 |= r3;   \
    r3 ^= r1;   \
    r1 &= r4;   \
    r4 ^= r2;   \
    r2 ^= r3;   \
    r3 &= r0;   \
    r4 |= r1;   \
    r3 ^= r4;   \
    r0 ^= r1;   \
    r4 &= r0;   \
    r1 ^= r3;   \
    r4 ^= r2;   \
    r1 |= r0;   \
    r1 ^= r2;   \
    r0 ^= r3;   \
    r2 = r1;   \
    r1 |= r3;   \
    r1 ^= r0;   \
	    }

#define I3(i, r0, r1, r2, r3, r4) \
       {	   \
    r4 = r2;   \
    r2 ^= r1;   \
    r1 &= r2;   \
    r1 ^= r0;   \
    r0 &= r4;   \
    r4 ^= r3;   \
    r3 |= r1;   \
    r3 ^= r2;   \
    r0 ^= r4;   \
    r2 ^= r0;   \
    r0 |= r3;   \
    r0 ^= r1;   \
    r4 ^= r2;   \
    r2 &= r3;   \
    r1 |= r3;   \
    r1 ^= r2;   \
    r4 ^= r0;   \
    r2 ^= r4;   \
	    }

#define S4(i, r0, r1, r2, r3, r4) \
       {	   \
    r1 ^= r3;   \
    r3 = ~r3;      \
    r2 ^= r3;   \
    r3 ^= r0;   \
    r4 = r1;   \
    r1 &= r3;   \
    r1 ^= r2;   \
    r4 ^= r3;   \
    r0 ^= r4;   \
    r2 &= r4;   \
    r2 ^= r0;   \
    r0 &= r1;   \
    r3 ^= r0;   \
    r4 |= r1;   \
    r4 ^= r0;   \
    r0 |= r3;   \
    r0 ^= r2;   \
    r2 &= r3;   \
    r0 = ~r0;      \
    r4 ^= r2;   \
	    }

#define I4(i, r0, r1, r2, r3, r4) \
       {	   \
    r4 = r2;   \
    r2 &= r3;   \
    r2 ^= r1;   \
    r1 |= r3;   \
    r1 &= r0;   \
    r4 ^= r2;   \
    r4 ^= r1;   \
    r1 &= r2;   \
    r0 = ~r0;      \
    r3 ^= r4;   \
    r1 ^= r3;   \
    r3 &= r0;   \
    r3 ^= r2;   \
    r0 ^= r1;   \
    r2 &= r0;   \
    r3 ^= r0;   \
    r2 ^= r4;   \
    r2 |= r3;   \
    r3 ^= r0;   \
    r2 ^= r1;   \
	    }

#define S5(i, r0, r1, r2, r3, r4) \
       {	   \
    r0 ^= r1;   \
    r1 ^= r3;   \
    r3 = ~r3;      \
    r4 = r1;   \
    r1 &= r0;   \
    r2 ^= r3;   \
    r1 ^= r2;   \
    r2 |= r4;   \
    r4 ^= r3;   \
    r3 &= r1;   \
    r3 ^= r0;   \
    r4 ^= r1;   \
    r4 ^= r2;   \
    r2 ^= r0;   \
    r0 &= r3;   \
    r2 = ~r2;      \
    r0 ^= r4;   \
    r4 |= r3;   \
    r2 ^= r4;   \
	    }

#define I5(i, r0, r1, r2, r3, r4) \
       {	   \
    r1 = ~r1;      \
    r4 = r3;   \
    r2 ^= r1;   \
    r3 |= r0;   \
    r3 ^= r2;   \
    r2 |= r1;   \
    r2 &= r0;   \
    r4 ^= r3;   \
    r2 ^= r4;   \
    r4 |= r0;   \
    r4 ^= r1;   \
    r1 &= r2;   \
    r1 ^= r3;   \
    r4 ^= r2;   \
    r3 &= r4;   \
    r4 ^= r1;   \
    r3 ^= r0;   \
    r3 ^= r4;   \
    r4 = ~r4;      \
	    }


#define S6(i, r0, r1, r2, r3, r4) \
       {	   \
    r2 = ~r2;      \
    r4 = r3;   \
    r3 &= r0;   \
    r0 ^= r4;   \
    r3 ^= r2;   \
    r2 |= r4;   \
    r1 ^= r3;   \
    r2 ^= r0;   \
    r0 |= r1;   \
    r2 ^= r1;   \
    r4 ^= r0;   \
    r0 |= r3;   \
    r0 ^= r2;   \
    r4 ^= r3;   \
    r4 ^= r0;   \
    r3 = ~r3;      \
    r2 &= r4;   \
    r2 ^= r3;   \
	    }

#define I6(i, r0, r1, r2, r3, r4) \
       {	   \
    r0 ^= r2;   \
    r4 = r2;   \
    r2 &= r0;   \
    r4 ^= r3;   \
    r2 = ~r2;      \
    r3 ^= r1;   \
    r2 ^= r3;   \
    r4 |= r0;   \
    r0 ^= r2;   \
    r3 ^= r4;   \
    r4 ^= r1;   \
    r1 &= r3;   \
    r1 ^= r0;   \
    r0 ^= r3;   \
    r0 |= r2;   \
    r3 ^= r1;   \
    r4 ^= r0;   \
	    }

#define S7(i, r0, r1, r2, r3, r4) \
       {	   \
    r4 = r2;   \
    r2 &= r1;   \
    r2 ^= r3;   \
    r3 &= r1;   \
    r4 ^= r2;   \
    r2 ^= r1;   \
    r1 ^= r0;   \
    r0 |= r4;   \
    r0 ^= r2;   \
    r3 ^= r1;   \
    r2 ^= r3;   \
    r3 &= r0;   \
    r3 ^= r4;   \
    r4 ^= r2;   \
    r2 &= r0;   \
    r4 = ~r4;      \
    r2 ^= r4;   \
    r4 &= r0;   \
    r1 ^= r3;   \
    r4 ^= r1;   \
	    }

#define I7(i, r0, r1, r2, r3, r4) \
       {	   \
    r4 = r2;   \
    r2 ^= r0;   \
    r0 &= r3;   \
    r2 = ~r2;      \
    r4 |= r3;   \
    r3 ^= r1;   \
    r1 |= r0;   \
    r0 ^= r2;   \
    r2 &= r4;   \
    r1 ^= r2;   \
    r2 ^= r0;   \
    r0 |= r2;   \
    r3 &= r4;   \
    r0 ^= r3;   \
    r4 ^= r1;   \
    r3 ^= r4;   \
    r4 |= r0;   \
    r3 ^= r2;   \
    r4 ^= r2;   \
	    }

// key xor
#define KX(r, a, b, c, d, e)    {\
	a ^= subkey[4 * r + 0]; \
	b ^= subkey[4 * r + 1]; \
	c ^= subkey[4 * r + 2]; \
	d ^= subkey[4 * r + 3];}


exception_t* serpent(serpent_key* user_key, block128* blocks, int block_count, enum mode mode, enum encryption encryption) {
	char* function_name = "serpent()";
	exception_t* exception;

	// Validate parameters.
	if ( user_key == NULL ) {
		return exception_throw("user_key was NULL.", function_name);
	}
	if ( blocks == NULL ) {
		return exception_throw("blocks was NULL.", function_name);
	}

	// Run the appropirate algorithm.
	switch(mode) {
	case CUDA:
		switch(encryption) {
		case DECRYPT:
			exception = serpent_cuda_decrypt(user_key, blocks, block_count);
			break;
		case ENCRYPT:
			exception = serpent_cuda_encrypt(user_key, blocks, block_count);
			break;
		default:
			return exception_throw("Unrecognized encryption parameter for CUDA.", function_name);
		}
		break;
	case PARALLEL:
		switch(encryption) {
		case DECRYPT:
			exception = serpent_parallel_decrypt(user_key, blocks, block_count);
			break;
		case ENCRYPT:
			exception = serpent_parallel_encrypt(user_key, blocks, block_count);
			break;
		default:
			return exception_throw("Unrecognized encryption parameter for parallel.", function_name);
		}
		break;
	case SERIAL:
		switch(encryption) {
		case DECRYPT:
			exception = serpent_serial_decrypt(user_key, blocks, block_count);
			break;
		case ENCRYPT:
			exception = serpent_serial_encrypt(user_key, blocks, block_count);
			break;
		default:
			return exception_throw("Unrecognized encryption parameter for serial.", function_name);
		}
		break;
	default:
		return exception_throw("Unknown mode.", function_name);
	}
	if ( exception != NULL ) {
		return exception_append(exception, function_name);
	}

	// Return success.
	return NULL;
}


exception_t* serpent_cuda_decrypt(serpent_key* user_key, block128* blocks, int block_count) {
	char* function_name = "serpent_cuda_decrypt()";
	
	return exception_throw("Not implemented.", function_name);
}


exception_t* serpent_cuda_encrypt(serpent_key* user_key, block128* blocks, int block_count) {
	char* function_name = "serpent_cuda_encrypt()";
	exception_t* exception;
	uint32* subkey;

	// Validate parameters.
	if ( user_key == NULL ) {
		return exception_throw("user_key was NULL.", function_name);
	}
	if ( blocks == NULL ) {
		return exception_throw("blocks was NULL.", function_name);
	}

	// Initialize the subkey.
	exception = serpent_init_subkey(user_key, &subkey);
	if ( exception != NULL ) {
		return exception_append(exception, function_name);
	}

	// Run the encryption algorithm from a different file.
	if ( serpent_cuda_encrypt_cu(subkey, blocks, block_count) == -1 ) {
		return exception_throw("CUDA encryption FAILED.", function_name);
	}

	// Free the subkey.
	exception = serpent_free_subkey(subkey);
	if ( exception != NULL ) {
		return exception_append(exception, function_name);
	}

	// Return success.
	return NULL;
}


exception_t* serpent_parallel_decrypt(serpent_key* user_key, block128* blocks, int block_count) {
	char* function_name = "serpent_parallel_decrypt()";

	return exception_throw("Not implemented.", function_name);
}


exception_t* serpent_parallel_encrypt(serpent_key* user_key, block128* blocks, int block_count) {
	char* function_name = "serpent_parallel_encrypt()";

	return exception_throw("Not implemented.", function_name);
}


exception_t* serpent_serial_decrypt(serpent_key* user_key, block128* blocks, int block_count) {
	char* function_name = "serpent_serial_decrypt()";
	exception_t* exception;
	uint32* subkey;
	uint32 a, b, c, d, e;

	// Validate parameters.
	if ( user_key == NULL ) {
		return exception_throw("user_key was NULL.", function_name);
	}
	if ( blocks == NULL ) {
		return exception_throw("blocks was NULL.", function_name);
	}

	// Initialize the subkey.
	exception = serpent_init_subkey(user_key, &subkey);
	if ( exception != NULL ) {
		return exception_append(exception, function_name);
	}

	// Decrypt each block.
	for ( int i = 0; i < block_count; i++ ) {
		block128* current_block;
		int j;

		// Get the current block.
		current_block = &(blocks[i]);

		// Change to little endian.
		exception = mirror_bytes32(current_block->x0, &a);
                if ( exception != NULL ) {
                        return exception_throw("Unable to mirror bytes to a.", function_name);
                }
                exception = mirror_bytes32(current_block->x1, &b);
                if ( exception != NULL ) {
                        return exception_throw("Unable to mirror bytes to b.", function_name);
                }
                exception = mirror_bytes32(current_block->x2, &c);
                if ( exception != NULL ) {
                        return exception_throw("Unable to mirror bytes to c.", function_name);
                }
                exception = mirror_bytes32(current_block->x3, &d);
                if ( exception != NULL ) {
                        return exception_throw("Unable to mirror bytes to d.", function_name);
                }

		// Decrypt the current block.
		j = 4;
		subkey += 96;
		beforeI7(KX);
		goto start;
		do
		{
			c = b;
			b = d;
			d = e;
			subkey -= 32;
			beforeI7(inverse_linear_transformation);
		start:
			beforeI7(I7); afterI7(KX);
			afterI7(inverse_linear_transformation); afterI7(I6); afterI6(KX);
			afterI6(inverse_linear_transformation); afterI6(I5); afterI5(KX);
			afterI5(inverse_linear_transformation); afterI5(I4); afterI4(KX);
			afterI4(inverse_linear_transformation); afterI4(I3); afterI3(KX);
			afterI3(inverse_linear_transformation); afterI3(I2); afterI2(KX);
			afterI2(inverse_linear_transformation); afterI2(I1); afterI1(KX);
			afterI1(inverse_linear_transformation); afterI1(I0); afterI0(KX);
		}
		while (--j != 0);

		// Restore to big endian, taking into account the significance of each block.
                exception = mirror_bytes32(a, &(current_block->x0));
                if ( exception != NULL ) {
                        return exception_throw("Unable to mirror bytes from a.", function_name);
                }
                exception = mirror_bytes32(d, &(current_block->x1));
                if ( exception != NULL ) {
                        return exception_throw("Unable to mirror bytes from d.", function_name);
                }
                exception = mirror_bytes32(b, &(current_block->x2));
                if ( exception != NULL ) {
                        return exception_throw("Unable to mirror bytes from b.", function_name);
                }
                exception = mirror_bytes32(e, &(current_block->x3));
                if ( exception != NULL ) {
                        return exception_throw("Unable to mirror bytes from e.", function_name);
                }
	}

	// Free the subkey.
	exception = serpent_free_subkey(subkey);
	if ( exception != NULL ) {
		return exception_append(exception, function_name);
	}

	// Return success.
	return NULL;
}


exception_t* serpent_serial_encrypt(serpent_key* user_key, block128* blocks, int block_count) {
	char* function_name = "serpent_serial_encrypt()";
	exception_t* exception;
	uint32* subkey;
	uint32 a, b, c, d, e;

	// Validate parameters.
	if ( user_key == NULL ) {
		return exception_throw("user_key was NULL.", function_name);
	}
	if ( blocks == NULL ) {
		return exception_throw("blocks was NULL.", function_name);
	}

	// Initialize the subkey.
	exception = serpent_init_subkey(user_key, &subkey);
	if ( exception != NULL ) {
		return exception_append(exception, function_name);
	}

	// Encrypt each block.
	for ( int i = 0; i < block_count; i++ ) {
		int j;

		// Get the current block.
		block128* current_block = &(blocks[i]);

		// Change to little endian.
		exception = mirror_bytes32(current_block->x0, &a);
		if ( exception != NULL ) {
			return exception_throw("Unable to mirror bytes to a.", function_name);
		}
		exception = mirror_bytes32(current_block->x1, &b);
		if ( exception != NULL ) {
			return exception_throw("Unable to mirror bytes to b.", function_name);
		}
		exception = mirror_bytes32(current_block->x2, &c);
		if ( exception != NULL ) {
			return exception_throw("Unable to mirror bytes to c.", function_name);
		}
		exception = mirror_bytes32(current_block->x3, &d);
		if ( exception != NULL ) {
			return exception_throw("Unable to mirror bytes to d.", function_name);
		}

		// Do the actual encryption.
		j = 1;
		do {
			beforeS0(KX); beforeS0(S0); afterS0(linear_transformation);
			afterS0(KX); afterS0(S1); afterS1(linear_transformation);
			afterS1(KX); afterS1(S2); afterS2(linear_transformation);
			afterS2(KX); afterS2(S3); afterS3(linear_transformation);
			afterS3(KX); afterS3(S4); afterS4(linear_transformation);
			afterS4(KX); afterS4(S5); afterS5(linear_transformation);
			afterS5(KX); afterS5(S6); afterS6(linear_transformation);
			afterS6(KX); afterS6(S7);

			if (j == 4)
				break;

			++j;
			c = b;
			b = e;
			e = d;
			d = a;
			a = e;
			subkey += 32;
			beforeS0(linear_transformation);
		} while (1);
		afterS7(KX);
		subkey -= 96;

		// Restore to big endian.
		exception = mirror_bytes32(d, &(current_block->x0));
		if ( exception != NULL ) {
			return exception_throw("Unable to mirror bytes from d.", function_name);
		}
		exception = mirror_bytes32(e, &(current_block->x1));
		if ( exception != NULL ) {
			return exception_throw("Unable to mirror bytes from e.", function_name);
		}
		exception = mirror_bytes32(b, &(current_block->x2));
		if ( exception != NULL ) {
			return exception_throw("Unable to mirror bytes from b.", function_name);
		}
		exception = mirror_bytes32(a, &(current_block->x3));
		if ( exception != NULL ) {
			return exception_throw("Unable to mirror bytes from a.", function_name);
		}
	}

	// Free the subkey.
	exception = serpent_free_subkey(subkey);
	if ( exception != NULL ) {
		return exception_append(exception, function_name);
	}

	// Return success.
	return NULL;
}


exception_t* serpent_init_subkey(serpent_key* user_key, uint32** subkey) {
	char* function_name = "serpent_init_subkey()";
	const int PREKEY_SIZE = 140;
	exception_t* exception;
	uint32* genkey;
	uint32 a, b, c, d, e;

	// Validate parameters.
	if ( user_key == NULL ) {
		return exception_throw("user_key was NULL.", function_name);
	}
	if ( subkey == NULL ) {
		return exception_throw("subkey was NULL.", function_name);
	}

	// Allocate space for genkey.
	genkey = (uint32*)malloc(sizeof(uint32) * PREKEY_SIZE);
	if ( genkey == NULL ) {
		return exception_throw("Unable to allocate genkey.", function_name);
	}

	// Assign user_key to the genkey; making sure to properly little-endianized the user key.
	for ( int i = 0; i < 8; i++ ) {
		uint32 word;
		// Get the key value.
		exception = serpent_key_get_word(user_key, i, &word);
		if ( exception != NULL ) {
			return exception_append(exception, function_name);
		}

		// Endianize the key value.
		exception = mirror_bytes32(word, &word);
		if ( exception != NULL ) {
			return exception_append(exception, function_name);
		}

		// Set the key value.
		genkey[i] = word;
	}

	// Generate the prekey by the following affine recurrence.
	genkey += 8;
	uint32 t = genkey[-1];
	for ( int i = 0; i < 132; ++i ) {
		genkey[i] = t = rotl_fixed(genkey[i-8] ^ genkey[i-5] ^ genkey[i-3] ^ t ^ 0x9E3779B9 ^ i, 11);
	}
	genkey -= 20;

#define LK(r, a, b, c, d, e)    {\
	a = genkey[(8-r)*4 + 0];	     \
	b = genkey[(8-r)*4 + 1];	     \
	c = genkey[(8-r)*4 + 2];	     \
	d = genkey[(8-r)*4 + 3];}

#define SK(r, a, b, c, d, e)    {\
	genkey[(8-r)*4 + 4] = a;	     \
	genkey[(8-r)*4 + 5] = b;	     \
	genkey[(8-r)*4 + 6] = c;	     \
	genkey[(8-r)*4 + 7] = d;}    \

	// Generare the subkey using the prekey.
	for ( int i = 0; i < 4 ; i++ )
	{
		afterS2(LK); afterS2(S3); afterS3(SK);
		afterS1(LK); afterS1(S2); afterS2(SK);
		afterS0(LK); afterS0(S1); afterS1(SK);
		beforeS0(LK); beforeS0(S0); afterS0(SK);
		genkey += 8*4;
		afterS6(LK); afterS6(S7); afterS7(SK);
		afterS5(LK); afterS5(S6); afterS6(SK);
		afterS4(LK); afterS4(S5); afterS5(SK);
		afterS3(LK); afterS3(S4); afterS4(SK);
	}
	afterS2(LK); afterS2(S3); afterS3(SK);
	genkey -= 108;

	// Assign output parameter.
	(*subkey) = genkey;

	// Return success.
	return NULL;
}


exception_t* serpent_free_subkey(uint32* subkey) {
	char* function_name = "serpent_free_subkey()";

	// Validate parameters.
	if ( subkey == NULL ) {
		return exception_throw("subkey was NULL.", function_name);
	}

	// Free the subkey.
	// Note that the original key be freed, too.
	free(subkey-8);

	// Return success.
	return NULL;
}


exception_t* serpent_key_get_word(serpent_key* key, int index, uint32* word) {
	char* function_name = "serpent_key_get_word()";

	// Validate parameters.
	if ( key == NULL ) {
		return exception_throw("key was NULL.", function_name);
	}
	if ( word == NULL ) {
		return exception_throw("word was NULL.", function_name);
	}
	if ( index < 0 || index > 7 ) {
		return exception_throw("Invalid index.", function_name);
	}

	// Set output parameter.
	switch(index) {
	case 0:
		(*word) = key->key0.x0;
		break;
	case 1:
		(*word) = key->key0.x1;
		break;
	case 2:
		(*word) = key->key0.x2;
		break;
	case 3:
		(*word) = key->key0.x3;
		break;
	case 4:
		(*word) = key->key1.x0;
		break;
	case 5:
		(*word) = key->key1.x1;
		break;
	case 6:
		(*word) = key->key1.x2;
		break;
	case 7:
		(*word) = key->key1.x3;
		break;
	default:
		return exception_throw("Unknown index.", function_name);
	}

	// Return success.
	return NULL;
}


exception_t* serpent_key_set_word(serpent_key* key, int index, uint32 word) {
	char* function_name = "serpent_key_set_word()";

	// Validate parameters.
	if ( key == NULL ) {
		return exception_throw("key was NULL.", function_name);
	}
	if ( index < 0 || index > 7 ) {
		return exception_throw("Invalid index.", function_name);
	}

	// Set output parameter.
	switch(index) {
	case 0:
		key->key0.x0 = word;
		break;
	case 1:
		key->key0.x1 = word;
		break;
	case 2:
		key->key0.x2 = word;
		break;
	case 3:
		key->key0.x3 = word;
		break;
	case 4:
		key->key1.x0 = word;
		break;
	case 5:
		key->key1.x1 = word;
		break;
	case 6:
		key->key1.x2 = word;
		break;
	case 7:
		key->key1.x3 = word;
		break;
	default:
		return exception_throw("Unknown index.", function_name);
	}

	// Return success.
	return NULL;
}
