// See aes.h for license terms and function comments.


#include "aes.h"


exception_t* aes(key256_t* key, block128_t* blocks, int block_count, enum mode mode, enum encryption encryption) {
	char* function_name = "aes()";
	exception_t* exception;

	// Validate parameters.
	if ( key == NULL ) {
		return exception_throw("key was NULL.", function_name);
	}
	else if ( blocks == NULL ) {
		return exception_throw("blocks was NULL.", function_name);
	}

	// Run the appropriate algorithm.
	switch(mode) {
	case CUDA:
		switch(encryption) {
		case DECRYPT:
			exception = aes_cuda_decrypt(key, blocks, block_count);
			break;
		case ENCRYPT:
			exception = aes_cuda_encrypt(key, blocks, block_count);
			break;
		default:
			return exception_throw("Unrecognized encryption parameter for CUDA.", function_name);
		}
		break;
	case PARALLEL:
		switch(encryption) {
		case DECRYPT:
			exception = aes_parallel_decrypt(key, blocks, block_count);
			break;
		case ENCRYPT:
			exception = aes_parallel_encrypt(key, blocks, block_count);
			break;
		default:
			return exception_throw("Unrecognized encryption parameter for parallel.", function_name);
		}
		break;
	case SERIAL:
		switch(encryption) {
		case DECRYPT:
			exception = aes_serial_decrypt(key, blocks, block_count);
			break;
		case ENCRYPT:
			exception = aes_serial_encrypt(key, blocks, block_count);
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


exception_t* aes_cuda_decrypt(key256_t* key, block128_t* blocks, int block_count) {
	char* function_name = "aes_cuda_decrypt()";

	return exception_throw("Not implemented.", function_name);
}


exception_t* aes_cuda_encrypt(key256_t* key, block128_t* blocks, int block_count) { 
	char* function_name = "aes_cuda_encrypt()";

	return exception_throw("Not implemented.", function_name);
}


exception_t* aes_parallel_decrypt(key256_t* key, block128_t* blocks, int block_count) {
	char* function_name = "aes_parallel_decrypt()";

	return exception_throw("Not implemented.", function_name);
}


exception_t* aes_parallel_encrypt(key256_t* key, block128_t* blocks, int block_count) {
	char* function_name = "aes_parallel_encrypt()";

	return exception_throw("Not implemented.", function_name);
}


exception_t* aes_serial_decrypt(key256_t* key, block128_t* blocks, int block_count) {
	char* function_name = "aes_serial_decrypt()";

	return exception_throw("Not implemented.", function_name);
}


exception_t* aes_serial_encrypt(key256_t* key, block128_t* blocks, int block_count) { 
	char* function_name = "aes_serial_encrypt()";

	return exception_throw("Not implemented.", function_name);
}

