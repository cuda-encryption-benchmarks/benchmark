#include "block128.h"


exception_t* block128_read(int fd, block128_t* block) {
	char* function_name = "block128_read";
	uint32_t buffer;

	// Validate parameters.
	if ( block == NULL ) {
		return exception_throw("block was NULL.", function_name);
	}

	// Read data in from the file.
	if ( read(fd, &buffer, sizeof(buffer)) == -1 ) {
		perror(NULL);
		return exception_throw("Read 1 FAILED.", function_name);
	}
	block->x0 = buffer;
	if ( read(fd, &buffer, sizeof(buffer)) == -1 ) {
		perror(NULL);
		return exception_throw("Read 2 FAILED.", function_name);
	}
	block->x1 = buffer;
	if ( read(fd, &buffer, sizeof(buffer)) == -1 ) {
		perror(NULL);
		return exception_throw("Read 3 FAILED.", function_name);
	}
	block->x2 = buffer;
	if ( read(fd, &buffer, sizeof(buffer)) == -1 ) {
		perror(NULL);
		return exception_throw("Read 4 FAILED.", function_name);
	}
	block->x3 = buffer;

	// Return success.
	return NULL;
}


exception_t* block128_write(int fd, block128_t* block) {
	char* function_name = "block128_write";

	// Validate parameters.
	if ( block == NULL ) {
		return exception_throw("block was NULL.", function_name);
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


