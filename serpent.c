// See serpent.h

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


// The nstruction sequences for the S-box functions
// come from Dag Arne Osvik's paper "Speeding Up Serpent".
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


exception_t* serpent(key256_t* user_key, block128_t* blocks, int block_count, enum mode mode, enum encryption encryption, size_t* buffer_size) {
	char* function_name = "serpent()";
	exception_t* exception;

	// Validate parameters.
	#ifdef DEBUG_SERPENT
	if ( user_key == NULL ) {
		return exception_throw("user_key was NULL.", function_name);
	}
	if ( blocks == NULL ) {
		return exception_throw("blocks was NULL.", function_name);
	}
	if ( buffer_size == NULL ) {
		return exception_throw("buffer_size was NULL.", function_name);
	}
	#endif

	// Run the appropirate algorithm.
	switch(mode) {
	case CUDA:
		switch(encryption) {
		case DECRYPT:
			exception = serpent_cuda_decrypt(user_key, blocks, block_count, buffer_size);
			break;
		case ENCRYPT:
			exception = serpent_cuda_encrypt(user_key, blocks, block_count, buffer_size);
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


exception_t* serpent_cuda_decrypt(key256_t* user_key, block128_t* blocks, int block_count, size_t* buffer_size) {
	char* function_name = "serpent_cuda_decrypt()";
	exception_t* exception;
	uint32_t* subkey;

	// Validate parameters.
	#ifdef DEBUG_SERPENT
	if ( user_key == NULL ) {
		return exception_throw("user_key was NULL.", function_name);
	}
	if ( blocks == NULL ) {
		return exception_throw("blocks was NULL.", function_name);
	}
	if ( buffer_size == NULL ) {
		return exception_throw("buffer_size was NULL.", function_name);
	}
	#endif

	// Initialize the subkey.
	exception = serpent_init_subkey(user_key, &subkey);
	if ( exception != NULL ) {
		return exception_append(exception, function_name);
	}

	// Run the encryption algorithm from a different file.
	if ( serpent_cuda_decrypt_cu(subkey, blocks, block_count, buffer_size) == -1 ) {
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


exception_t* serpent_cuda_encrypt(key256_t* user_key, block128_t* blocks, int block_count, size_t* buffer_size) {
	char* function_name = "serpent_cuda_encrypt()";
	exception_t* exception;
	uint32_t* subkey;

	// Validate parameters.
	#ifdef DEBUG_SERPENT
	if ( user_key == NULL ) {
		return exception_throw("user_key was NULL.", function_name);
	}
	if ( blocks == NULL ) {
		return exception_throw("blocks was NULL.", function_name);
	}
	if ( buffer_size == NULL ) {
		return exception_throw("buffer_size was NULL.", function_name);
	}
	#endif

	// Initialize the subkey.
	exception = serpent_init_subkey(user_key, &subkey);
	if ( exception != NULL ) {
		return exception_append(exception, function_name);
	}

	// Run the encryption algorithm from a different file.
	if ( serpent_cuda_encrypt_cu(subkey, blocks, block_count, buffer_size) == -1 ) {
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


void serpent_decrypt_block(block128_t* block, uint32_t* subkey) {
	//char* function_name = "serpent_decrypt_block()";
	uint32_t a, b, c, d, e;
	int j;

	// Change to little endian.
	a = mirror_bytes32(block->x0);
	b = mirror_bytes32(block->x1);
	c = mirror_bytes32(block->x2);
	d = mirror_bytes32(block->x3);

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
	block->x0 = mirror_bytes32(a);
	block->x1 = mirror_bytes32(d);
	block->x2 = mirror_bytes32(b);
	block->x3 = mirror_bytes32(e);
}


void serpent_encrypt_block(block128_t* block, uint32_t* subkey) {
	//char* function_name = "serpent_encrypt_block()";
	//exception_t* exception;
	uint32_t a, b, c, d, e;
	int j;

	// Change to little endian.
	a = mirror_bytes32(block->x0);
	b = mirror_bytes32(block->x1);
	c = mirror_bytes32(block->x2);
	d = mirror_bytes32(block->x3);

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

	// Restore to big endian.
	block->x0 = mirror_bytes32(d);
	block->x1 = mirror_bytes32(e);
	block->x2 = mirror_bytes32(b);
	block->x3 = mirror_bytes32(a);
}


exception_t* serpent_parallel_decrypt(key256_t* user_key, block128_t* blocks, int block_count) {
	char* function_name = "serpent_parallel_decrypt()";
	exception_t* exception;
	uint32_t* subkey;
	int blocks_per_thread;
	int thread_count;
	int thread_index;

	// Validate parameters.
	#ifdef DEBUG_SERPENT
	if ( user_key == NULL ) {
		return exception_throw("user_key was NULL.", function_name);
	}
	if ( blocks == NULL ) {
		return exception_throw("blocks was NULL.", function_name);
	}
	#endif

	// Initialize the subkey.
	exception = serpent_init_subkey(user_key, &subkey);
	if ( exception != NULL ) {
		return exception_append(exception, function_name);
	}

	// Decrypt each block.
	#pragma omp parallel shared(blocks, subkey, block_count, thread_count, thread_index, blocks_per_thread)
	{
		#if defined (_OPENMP)
			thread_count = omp_get_num_threads();
			thread_index = omp_get_thread_num();
		#endif

		// Calculate the current index.
		int index = thread_index;
		
		// Encrypted the minimal number of blocks.
		blocks_per_thread = block_count / thread_count;
		for ( int i = 0; i < blocks_per_thread; i++ ) {
			serpent_decrypt_block(&(blocks[index]), subkey);
			index += thread_count;
		}

		// Encrypt the extra blocks that fall outside the minimal number of block.s
		if ( index < block_count ) {
			serpent_decrypt_block(&(blocks[index]), subkey);
		}
	} 

	// Free the subkey.
	exception = serpent_free_subkey(subkey);
	if ( exception != NULL ) {
		return exception_append(exception, function_name);
	}

	// Return. 
	return NULL;
}


exception_t* serpent_parallel_encrypt(key256_t* user_key, block128_t* blocks, int block_count) {
	char* function_name = "serpent_parallel_encrypt()";
	exception_t* exception;
	uint32_t* subkey;
	int blocks_per_thread;
	int thread_count;
	int thread_index;

	// Validate parameters.
	#ifdef DEBUG_SERPENT
	if ( user_key == NULL ) {
		return exception_throw("user_key was NULL.", function_name);
	}
	if ( blocks == NULL ) {
		return exception_throw("blocks was NULL.", function_name);
	}
	#endif

	// Initialize the subkey.
	exception = serpent_init_subkey(user_key, &subkey);
	if ( exception != NULL ) {
		return exception_append(exception, function_name);
	}

	// Encrypt each block.
	#pragma omp parallel shared(blocks, subkey, block_count, thread_count, thread_index, blocks_per_thread)
	{
		#if defined (_OPENMP)
			thread_count = omp_get_num_threads();
			thread_index = omp_get_thread_num();
		#endif

		// Calculate the current index.
		int index = thread_index;
		
		// Encrypted the minimal number of blocks.
		blocks_per_thread = block_count / thread_count;
		for ( int i = 0; i < blocks_per_thread; i++ ) {
			serpent_encrypt_block(&(blocks[index]), subkey);
			index += thread_count;
		}

		// Encrypt the extra blocks that fall outside the minimal number of block.s
		if ( index < block_count ) {
			serpent_encrypt_block(&(blocks[index]), subkey);
		}
	} 

	// Free the subkey.
	exception = serpent_free_subkey(subkey);
	if ( exception != NULL ) {
		return exception_append(exception, function_name);
	}

	// Return. 
	return NULL;
}


exception_t* serpent_serial_decrypt(key256_t* user_key, block128_t* blocks, int block_count) {
	char* function_name = "serpent_serial_decrypt()";
	exception_t* exception;
	uint32_t* subkey;

	// Validate parameters.
	#ifdef DEBUG_SERPENT
	if ( user_key == NULL ) {
		return exception_throw("user_key was NULL.", function_name);
	}
	if ( blocks == NULL ) {
		return exception_throw("blocks was NULL.", function_name);
	}
	#endif

	// Initialize the subkey.
	exception = serpent_init_subkey(user_key, &subkey);
	if ( exception != NULL ) {
		return exception_append(exception, function_name);
	}

	// Decrypt each block.
	for ( int i = 0; i < block_count; i++ ) {
		serpent_decrypt_block(&(blocks[i]), subkey);
	}

	// Free the subkey.
	exception = serpent_free_subkey(subkey);
	if ( exception != NULL ) {
		return exception_append(exception, function_name);
	}

	// Return success.
	return NULL;
}


exception_t* serpent_serial_encrypt(key256_t* user_key, block128_t* blocks, int block_count) {
	char* function_name = "serpent_serial_encrypt()";
	exception_t* exception;
	uint32_t* subkey;

	// Validate parameters.
	#ifdef DEBUG_SERPENT
	if ( user_key == NULL ) {
		return exception_throw("user_key was NULL.", function_name);
	}
	if ( blocks == NULL ) {
		return exception_throw("blocks was NULL.", function_name);
	}
	#endif

	// Initialize the subkey.
	exception = serpent_init_subkey(user_key, &subkey);
	if ( exception != NULL ) {
		return exception_append(exception, function_name);
	}

	// Encrypt each block.
	for ( int i = 0; i < block_count; i++ ) {
		serpent_encrypt_block(&(blocks[i]), subkey);
	}

	// Free the subkey.
	exception = serpent_free_subkey(subkey);
	if ( exception != NULL ) {
		return exception_append(exception, function_name);
	}

	// Return success.
	return NULL;
}


exception_t* serpent_init_subkey(key256_t* user_key, uint32_t** subkey) {
	char* function_name = "serpent_init_subkey()";
	const int PREKEY_SIZE = 140;
	exception_t* exception;
	uint32_t* genkey;
	uint32_t a, b, c, d, e;

	// Validate parameters.
	#ifdef DEBUG_SERPENT
	if ( user_key == NULL ) {
		return exception_throw("user_key was NULL.", function_name);
	}
	if ( subkey == NULL ) {
		return exception_throw("subkey was NULL.", function_name);
	}
	#endif

	// Allocate space for genkey.
	genkey = (uint32_t*)malloc(sizeof(uint32_t) * PREKEY_SIZE);
	if ( genkey == NULL ) {
		return exception_throw("Unable to allocate genkey.", function_name);
	}

	// Assign user_key to the genkey; making sure to properly little-endianized the user key.
	for ( int i = 0; i < 8; i++ ) {
		uint32_t word;
		// Get the key value.
		exception = key256_get_word(user_key, i, &word);
		if ( exception != NULL ) {
			return exception_append(exception, function_name);
		}

		// Endianize the key value.
		word = mirror_bytes32(word);

		// Set the key value.
		genkey[i] = word;
	}

	// Generate the prekey by the following affine recurrence.
	genkey += 8;
	uint32_t t = genkey[-1];
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


exception_t* serpent_free_subkey(uint32_t* subkey) {
	#ifdef DEBUG_SERPENT
	char* function_name = "serpent_free_subkey()";
	#endif

	// Validate parameters.
	#ifdef DEBUG_SERPENT
	if ( subkey == NULL ) {
		return exception_throw("subkey was NULL.", function_name);
	}
	#endif

	// Free the subkey.
	// Note that the original key be freed, too.
	free(subkey-8);

	// Return success.
	return NULL;
}


