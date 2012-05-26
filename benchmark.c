#include "benchmark.h"


exception_t* benchmark(key256_t* key, char* input_filepath, enum algorithm algorithm, enum mode mode, enum encryption encryption, benchmark_data_t* benchmark_data ) {
	char* function_name = "benchmark()";
	exception_t* exception;
	block128_t* blocks;
	file_t file;
	struct timespec clock_begin;
	struct timespec clock_elapsed;
	struct timespec clock_end;
	long long block_count;
	size_t buffer_size;

	// Validate parameters.
	if ( key == NULL ) {
		return exception_throw("key was NULL.", function_name);
	}
	if ( input_filepath == NULL ) {
		return exception_throw("input_filepath was NULL.", function_name);
	}
	if ( benchmark_data == NULL ) {
		return exception_throw("benchmark_data was NULL.", function_name);
	}

	// Open input file.
	if ( encryption == ENCRYPT ) {
		exception = file_init(input_filepath, UNENCRYPTED, &file);
	}
	else if ( encryption == DECRYPT ) {
		exception = file_init(input_filepath, ENCRYPTED, &file);
	}
	else { 
		return exception_throw("Unhandled encryption parameter.", function_name);
	}
	if ( exception != NULL ) {
		return exception_append(exception, function_name);
	}

	// Get number of blocks in file.
	exception = file_get_block_count(&file, &block_count);
	if ( exception != NULL ) {
		return exception_append(exception, function_name);
	}

	// Read data from file.
	fprintf(stdout, "Reading. ");
	fflush(stdout);
	exception = file_read(&file, 0, block_count, &blocks);
	if ( exception != NULL ) {
		return exception_append(exception, function_name);
	}

	// Print message for the user.
	if ( encryption == ENCRYPT ) {
		fprintf(stdout, "Encrypting. ");
	}
	else if ( encryption == DECRYPT ) {
		fprintf(stdout, "Decrypting. ");
	}
	else {
		return exception_throw("Invalid encryption parameter.", function_name);
	}
	fflush(stdout);

	// Get begin time.
	if ( clock_gettime(CLOCK_REALTIME, &clock_begin) == -1 ) {
		return exception_throw("Unable to get begin time.", function_name);
	}

	// Call the algorithm.
	switch(algorithm) {
	case AES:
		exception = aes(key, blocks, block_count, mode, encryption);
		break;
	case SERPENT:
		exception = serpent(key, blocks, block_count, mode, encryption, &buffer_size);
		break;
	case TWOFISH:
		exception = twofish(key, blocks, block_count, mode, encryption, &buffer_size);
		break;
	default:
		return exception_throw("Unrecognized algorithm.", function_name);
		break;
	}
	if ( exception != NULL ) {
		return exception_append(exception, function_name);
	}

	// Get end time.
	if ( clock_gettime(CLOCK_REALTIME, &clock_end) == -1 ) {
		return exception_throw("Unable to get end time.", function_name);
	}

	// Write data to file.
	fprintf(stdout, "Writing. ");
	fflush(stdout);
	exception = file_write(&file, 0, block_count, blocks);
	if ( exception != NULL ) {
		return exception_append(exception, function_name);
	}

	// Close input file.
	exception = file_free(&file);
	if ( exception != NULL ) {
		return exception_append(exception, function_name);
	}

	// Calculate execution time.
	clock_elapsed.tv_sec = clock_end.tv_sec - clock_begin.tv_sec;
	clock_elapsed.tv_nsec = clock_end.tv_nsec - clock_begin.tv_nsec;
	if ( clock_elapsed.tv_nsec < 0 ) {
		clock_elapsed.tv_nsec += 1000000000;
		clock_elapsed.tv_sec -= 1;
	}

	// Assign output parameters.
	benchmark_data->time_elapsed.tv_sec = clock_elapsed.tv_sec;
	benchmark_data->time_elapsed.tv_nsec = clock_elapsed.tv_nsec;
	benchmark_data->buffer_size = buffer_size;

	// Return success.
	return NULL;
}
