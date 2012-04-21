
#include "twofish.h"


union block128_u {
    block128_t data;
    uint32_t arr[4];
};

union key256_u {
    key256_t key;
    uint32_t arr[8];
};


exception_t* twofish_encrypt_block(block128_u* block, uint32_u* subkey){



}

exception_t* twofish_decrypt_block(block128_t* block, uint32_t* subkey){

}




exception_t* twofish(key256_t* user_key, block128_t* blocks, int block_count, enum mode mode, enum encryption encryption, size_t* buffer_size){
char* function_name = "twofish()";
	exception_t* exception;

	// Validate parameters.
	if ( user_key == NULL ) {
		return exception_throw("user_key was NULL.", function_name);
	}
	if ( blocks == NULL ) {
		return exception_throw("blocks was NULL.", function_name);
	}
	if ( buffer_size == NULL ) {
		return exception_throw("buffer_size was NULL.", function_name);
	}

	// Run the appropirate algorithm.
	switch(mode) {
	case CUDA:
		switch(encryption) {
		case DECRYPT:
			exception = twofish_cuda_decrypt(user_key, blocks, block_count, buffer_size);
			break;
		case ENCRYPT:
			exception = twofish_cuda_encrypt(user_key, blocks, block_count, buffer_size);
			break;
		default:
			return exception_throw("Unrecognized encryption parameter for CUDA.", function_name);
		}
		break;
	case PARALLEL:
		switch(encryption) {
		case DECRYPT:
			exception = twofish_parallel_decrypt(user_key, blocks, block_count);
			break;
		case ENCRYPT:
			exception = twofish_parallel_encrypt(user_key, blocks, block_count);
			break;
		default:
			return exception_throw("Unrecognized encryption parameter for parallel.", function_name);
		}
		break;
	case SERIAL:
		switch(encryption) {
		case DECRYPT:
			exception = twofish_serial_decrypt(user_key, blocks, block_count);
			break;
		case ENCRYPT:
			exception = twofish_serial_encrypt(user_key, blocks, block_count);
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



exception_t* twofish_serial_decrypt(key256_t* user_key, block128_t* blocks, int block_count) {
	char* function_name = "twofish_serial_decrypt()";
	exception_t* exception;
	uint32_t* subkey;
	//uint32_t a, b, c, d, e;

	// Validate parameters.
	if ( user_key == NULL ) {
		return exception_throw("user_key was NULL.", function_name);
	}
	if ( blocks == NULL ) {
		return exception_throw("blocks was NULL.", function_name);
	}

	// Initialize the subkey.
	exception = twofish_init_subkey(user_key, &subkey);
	if ( exception != NULL ) {
		return exception_append(exception, function_name);
	}

	// Decrypt each block.
	for ( int i = 0; i < block_count; i++ ) {
		exception = twofish_decrypt_block(&(blocks[i]), subkey);
		if ( exception != NULL ) {
			return exception_append(exception, function_name);
		}
	}

	// Free the subkey.
	exception = twofish_free_subkey(subkey);
	if ( exception != NULL ) {
		return exception_append(exception, function_name);
	}

	// Return success.
	return NULL;
}


exception_t* twofish_serial_encrypt(key256_t* user_key, block128_t* blocks, int block_count) {
	char* function_name = "twofish_serial_encrypt()";
	exception_t* exception;
	uint32_t* subkey;

	// Validate parameters.
	if ( user_key == NULL ) {
		return exception_throw("user_key was NULL.", function_name);
	}
	if ( blocks == NULL ) {
		return exception_throw("blocks was NULL.", function_name);
	}

	// Initialize the subkey.
	exception = twofish_init_subkey(user_key, &subkey);
	if ( exception != NULL ) {
		return exception_append(exception, function_name);
	}
    
    // Add the key struct to the union so that twofish can do indexing tricks
   key256_u key;
   key.key = *subkey;
    
    
	// Encrypt each block.
	for ( int i = 0; i < block_count; i++ ) {
        //Add the block to the union so that we can index it with array stuff
        block128_u block;
        block.data = blocks[i];
    
		exception = twofish_encrypt_block(&block, &key);
		if ( exception != NULL ) {
			return exception_throw("blocks was NULL.", function_name);
		}
	}

	// Free the subkey.
	exception = twofish_free_subkey(subkey);
	if ( exception != NULL ) {
		return exception_append(exception, function_name);
	}

	// Return success.
	return NULL;
}




exception_t* twofish_init_subkey(key256_t* user_key, uint32_t** subkey) {
	char* function_name = "twofish_init_subkey()";
	const int PREKEY_SIZE = 140;
	exception_t* exception;
	uint32_t* genkey;
	uint32_t a, b, c, d, e;

	// Validate parameters.
	if ( user_key == NULL ) {
		return exception_throw("user_key was NULL.", function_name);
	}
	if ( subkey == NULL ) {
		return exception_throw("subkey was NULL.", function_name);
	}

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


exception_t* twofish_free_subkey(uint32_t* subkey) {
	char* function_name = "twofish_free_subkey()";

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


exception_t* twofish_cuda_decrypt(key256_t* user_key, block128_t* blocks, int block_count, size_t* buffer_size) {
    return NULL;
}


/**	Encrypt the specified array of 128-bit blocks through the CUDA twofish algorithm.
 *	@return	NULL on success, exception_t* on failure.
 */
exception_t* twofish_cuda_encrypt(key256_t* user_key, block128_t* blocks, int block_count, size_t* buffer_size){
    return NULL;
}


/**	Private inner function to prevent linking errors because nvcc does not like external libraries.
 *	@return 0 on success, -1 on failure.
 */
int twofish_cuda_decrypt_cu(uint32_t* subkey, block128_t* blocks, int block_count, size_t* buffer_size){
    return NULL;

}



int twofish_cuda_encrypt_cu(uint32_t* subkey, block128_t* blocks, int block_count, size_t* buffer_size){
    return NULL;

}


/**	Decrypt the specified array of 128-bit blocks in parallel through the twofish encryption algorithm.
 *	@return NULL on success, exception_t* on failure.
 */
exception_t* twofish_parallel_decrypt(key256_t* user_key, block128_t* blocks, int block_count){
    return NULL;
}






/**	Encrypt the specified array of 128-bit blocks in parallel through the twofish encryption algorithm.
 *	@return NULL on success, exception_t* on failure.
 */
exception_t* twofish_parallel_encrypt(key256_t* user_key, block128_t* blocks, int block_count){
    return NULL;
}

