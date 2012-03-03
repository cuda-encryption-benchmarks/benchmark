#include "file.h"


exception_t* file_read_metadata(file_t* file) {
	char* function_name = "file_read_metadata()";
	long long temp;

	// Validate parameters.
	if ( file == NULL ) {
		return exception_throw("file was NULL.", function_name);
	}

	// Seek to end of the file.
	temp = lseek64(file->fd, 0, SEEK_END);
	if ( temp == -1 ) {
		perror(NULL);
		return exception_throw("Unable to lseek to file end.", function_name);
	}

	// Move back by the size of the metadata.
	temp = lseek64(file->fd, (int)(-sizeof(file->padding)), SEEK_CUR);
	if ( temp == -1 ) {
		perror(NULL);
		return exception_throw("Unable to lseek to metadata.", function_name);
	}

	// Read in metadata.
	temp = read(file->fd, &file->padding, sizeof(file->padding));
	if ( temp == -1 ) {
		perror(NULL);
		return exception_throw("Unable to read in metadata.", function_name);
	}

	// Delete the metadata.
	file->size -= sizeof(file->padding);
	if ( ftruncate(file->fd, (int)(file->size)) == -1 ) {
		perror(NULL);
		return exception_throw("Unable to truncate metadata.", function_name);
	}

	// Return success.
	return NULL;
}

exception_t* file_init(const char* file_name, enum file_encryption flag, file_t* file) { 
	char* function_name = "file_init()";
	exception_t* exception;
	struct stat64 buffer;
	int temp;

	// Validate parameters.
	if ( file_name == NULL ) {
		return exception_throw("file_name was NULL.", function_name);
	}
	if ( file == NULL ) {
		return exception_throw("file was NULL.", function_name);
	}

	// Initialize file structure.
	file->blocks = NULL;
	file->block_count = 0;
	file->padding = 0;
	file->flag = flag;

	// Get file name,
	file->name = (char*)malloc(sizeof(char)*strlen(file_name));
	if ( file->name == NULL ) {
		return exception_throw("Malloc failed.", function_name);
	}
	strcpy(file->name, file_name);

	// Open the file.
	file->fd = open(file_name, O_RDWR | O_LARGEFILE);
	if ( file->fd == -1 ) {
		perror("Unable to open file.");
		return exception_throw("Unable to open file.", function_name);
	}

	// Get file size.
	temp = fstat64(file->fd, &buffer);
	if ( temp == -1 ) {
		perror("Unable to get stats.");
		return exception_throw("Unable to get stats.", function_name);
	}
	file->size = buffer.st_size;

	// Get file metadata or pad the file.
	if ( flag == ENCRYPTED ) {
		// Read in file metadata.
		exception = file_read_metadata(file);
		if ( exception != NULL ) {
			return exception_append(exception, function_name);
		}
	}
	else if ( flag == UNENCRYPTED ) {
		int temp;

		// Calculate file padding.
		temp = file->size % sizeof(block128);
		if ( temp == 0 ) {
			file->padding = temp;
		}
		else {
			file->padding = sizeof(block128) - temp;

			// Pad the file.
			exception = file_write_padding(file);
			if ( exception != NULL ) {
				return exception_append(exception, function_name);
			}
		}
	}
	else {
		return exception_throw("Unexpected flag.", function_name);
	}

	// Return success.
	return NULL;
}


exception_t* file_free(file_t* file) {
	char* function_name = "file_free()";
	exception_t* exception;

	// Validate parameters.
	if ( file == NULL ) {
		return exception_throw("file was NULL.", function_name);
	}

	// Write metadata or truncate file.
	// Note that file->flag reflects the file when it was opened,
	// and the file is now the _opposite_ of what the flag states.
	if ( file->flag == UNENCRYPTED ) {
		// Write padding to the end of the file.
		exception = file_write_metadata(file);
		if ( exception != NULL ) {
			return exception_append(exception, function_name);
		}
	}
	else if ( file->flag == ENCRYPTED ) {
		// Truncate the file padding.
		if ( ftruncate(file->fd, (int)(file->size - file->padding)) == -1 ) {
			return exception_throw("Unable to truncate file.", function_name);
		}
	}
	else {
		return exception_throw("Unexpected flag.", function_name);
	}

	// Free file name,
	free(file->name);

	// Close the file.
	if ( close(file->fd) == -1 ) {
		perror("Unable to close file.");
		return exception_throw("Unable to close file.", function_name);
	}

	// Set structure defaults.
	file->blocks = NULL;
	file->name = NULL;
	file->block_count = 0;
	file->size = 0;
	file->padding = 0;
	file->fd = 0;
	file->flag = 0;

	// Return success.
	return NULL;
}


exception_t* file_get_block_count(file_t* file, long long* block_count) {
	char* function_name = "file_get_block_count()";

	// Validate parameters.
	if ( file == NULL ) {
		return exception_throw("file was NULL.", function_name);
	}
	if ( block_count == NULL ) {
		return exception_throw("block_count was NULL.", function_name);
	}

	// Calculate number of blocks.
	(*block_count) = file->size / sizeof(block128);

	// Return success.
	return NULL;
}


exception_t* file_read(file_t* file, int block_index, int block_count, block128** blocks, int* blocks_read) {
	char* function_name = "file_read()";
	exception_t* exception;
	// Temporary storage for blocks to prevent multiple dereferences.
	block128* blocks_local; 
	int blocks_read_local;

	// Validate parameters.
	if ( file == NULL ) {
		return exception_throw("file was NULL.", function_name);
	}
	if ( blocks == NULL ) {
		return exception_throw("blocks was NULL.", function_name);
	}
	if ( blocks_read == NULL ) {
		return exception_throw("blocks_read was NULL.", function_name);
	}

	// Allocate space for block_count block128s.
	blocks_local = (block128*)malloc(sizeof(block128) * block_count);
	if ( blocks_local == NULL ) {
		return exception_throw("Unable to allocate space for blocks.", function_name);
	}

	// Seek to position in the file at block_index.
	if ( lseek64(file->fd, (int)(sizeof(block128)*block_index), SEEK_SET) == -1 ) {
		return exception_throw("Unable to seek to block index.", function_name);
	}

	// Read in the blocks.
	blocks_read_local = 0;
	for ( int i = 0; i < block_count; i++ ) {
		exception = block128_read(file->fd, &(blocks_local[i]));
		if ( exception != NULL ) {
			return exception_append(exception, function_name);
		}
		blocks_read_local++;
	}

	// Set output parameters.
	(*blocks) = blocks_local;
	(*blocks_read) = blocks_read_local;

	// Return success.
	return NULL;
}


exception_t* file_write(file_t* file, int block_index, int block_count, block128* blocks, int* blocks_written) {
	char* function_name = "file_write()";
	int blocks_written_local;

	// Validate parameters.
	if ( file == NULL ) {
		return exception_throw("file was NULL.", function_name);
	}
	if ( blocks == NULL ) {
		return exception_throw("blocks was NULL.", function_name);
	}
	if ( blocks_written == NULL ) {
		return exception_throw("blocks_written was NULL.", function_name);
	}

	// Seek to block_index.
	if ( lseek64(file->fd, (int)(sizeof(block128)  * block_index), SEEK_SET) == -1 ) {
		perror(NULL);
		return exception_throw("Unable to seek to block_index.", function_name);
	}

	// Write blocks.
	blocks_written_local = 0;
	for ( int i = 0; i < block_count; i++ ) {
		if ( write(file->fd, &(blocks[i]), sizeof(block128)) == -1 ) {
			perror(NULL);
			return exception_throw("Unable to write block.", function_name);
		}
		blocks_written_local++;
	}

	// Set output parameters.
	(*blocks_written) = blocks_written_local;

	// Return success.
	return NULL;
}


exception_t* file_write_metadata(file_t* file) {
	char* function_name = "file_write_metadata()";

	// Validate parameters.
	if ( file == NULL ) {
		return exception_throw("file was NULL.", function_name);
	}

	// Seek to the end of the file.
	if ( lseek64(file->fd, 0, SEEK_END) == -1 ) {
		perror(NULL);
		return exception_throw("Unable to seek to end of file.", function_name);
	}

	// Write padding to end of file.
	if ( write(file->fd, &(file->padding), sizeof(file->padding)) == -1 ) {
		perror(NULL);
		return exception_throw("Unable to write metadata to file.", function_name);
	}

	// Return success.
	return NULL;
}


exception_t* file_write_padding(file_t* file) {
	char* function_name = "file_write_padding()";
	char buffer = '\0';

	// Validate parameters.
	if ( file == NULL ) {
		return exception_throw("file was NULL.", function_name);
	}
	if ( file->flag != UNENCRYPTED ) {
		return exception_throw("file must be unencrypted.", function_name);
	}

	// Seek to the end of the file.
	if ( lseek64(file->fd, 0, SEEK_END) == -1 ) {
		return exception_throw("Unable to seek to end of file.", function_name);
	}

	// Write padding to the end of the file.
	// This is a terrible way to do this, but it will work!
	file->size += file->padding;
	for ( int i = 0; i < file->padding; i++ ) {
		if ( write(file->fd, &buffer, sizeof(buffer)) == -1 ) {
			return exception_throw("Unable to write padding.", function_name);
		}
	}

	// Return success.
	return NULL;
}
