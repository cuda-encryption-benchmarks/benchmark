#include "block128.h"


exception_t* block128_read_block(int fd, block128* block) {
	char* function_name = "block128_read_block()";
	uint32 buffer;

	// Validate parameters.
	if ( block == NULL ) {
		return exception_throw("NULL parameter unacceptable.", function_name);
	}

	// Read data in from the file.
	if ( read(fd, &buffer, sizeof(buffer)) == -1 ) {
		perror(function_name);
		return exception_throw("Read 1 FAILED.", function_name);
	}
	block->x0 = buffer;
	if ( read(fd, &buffer, sizeof(buffer)) == -1 ) {
		perror(function_name);
		return exception_throw("Read 2 FAILED.", function_name);
	}
	block->x1 = buffer;
	if ( read(fd, &buffer, sizeof(buffer)) == -1 ) {
		perror(function_name);
		return exception_throw("Read 3 FAILED.", function_name);
	}
	block->x2 = buffer;
	if ( read(fd, &buffer, sizeof(buffer)) == -1 ) {
		perror(function_name);
		return exception_throw("Read 4 FAILED.", function_name);
	}
	block->x3 = buffer;

	// Return success.
	return NULL;
}


exception_t* block128_read_file(int fd, block128** blocks, int* block_count) {
	char* function_name = "block128_read_file()";
	exception_t* exception;
	long int file_size;
	struct stat file_stats;

	// Validate parameters.
	if ( blocks == NULL ) {
		return exception_throw("NULL blocks parameter unacceptable.", function_name);
	}
	if ( block_count == NULL ) {
		return exception_throw("NULL block_count parameter unacceptable.", function_name);
	}
	
	// Get file size.
	if ( fstat(fd, &file_stats) == -1 ) {
		perror(function_name);
		return exception_throw("Unable to retrieve file statistics.", function_name);
	}
	file_size = file_stats.st_size;

	// Allocate enough blocks for the file.
	(*block_count) = file_size / sizeof(block128);
	if ( (file_size % sizeof(block128)) > 0 ) {
		(*block_count)++;
	}
	(*blocks) = (block128*)malloc((*block_count) * sizeof(block128));

	// Read file into each block.
	for ( int i = 0; i < (*block_count); i++ ) {
		// Read in a single block.
		exception = block128_read_block(fd, (&(*blocks)[i]));
		if ( exception != NULL ) {
			return exception_append(exception, function_name);
		}
	}

	// Return success.
	return NULL;
}


exception_t* block128_write_block(int fd, block128* block) {
	char* function_name = "block128_write_block()";

	// Validate parameters.
	if ( block == NULL ) {
		return exception_throw("NULL parameter unacceptable.", function_name);
	}

	// Write data to the file.
	if ( write(fd, &block->x0, sizeof(block->x0)) == -1 ) {
		perror(function_name);
		return exception_throw("Write 1 FAILED.", function_name);
	}
	if ( write(fd, &block->x1, sizeof(block->x1)) == -1 ) {
		perror(function_name);
		return exception_throw("Write 2 FAILED.", function_name);
	}
	if ( write(fd, &block->x2, sizeof(block->x2)) == -1 ) {
		perror(function_name);
		return exception_throw("Write 3 FAILED.", function_name);
	}
	if ( write(fd, &block->x3, sizeof(block->x3)) == -1 ) {
		perror(function_name);
		return exception_throw("Write 4 FAILED.", function_name);
	}

	// Return success.
	return NULL;
}


exception_t* block128_write_file(int fd, block128* blocks, int block_count) {
	char* function_name = "block128_write_file()";
	exception_t* exception;

	// Validate parameters.
	if ( blocks == NULL ) {
		return exception_throw("NULL parameter unacceptable.", function_name);
	}

	// Write data to the file.
	for ( int i = 0; i < block_count; i++ ) {
		exception = block128_write_block(fd, &blocks[i]);
		if ( exception != NULL ) {
			return exception_append(exception, function_name);
		}
	}

	// Return success.
	return NULL;
}
