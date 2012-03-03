#ifndef file_H
#define file_H

// We have no idea why this works but it prevent an 'implicit declaration of ftruncate()'.
#define _BSD_SOURCE
// Open large files.
#define _LARGEFILE64_SOURCE

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#include "block128.h"
#include "ccc/ccc.h"


// Enumeration which specified whether the file to be read in is encrypted or not.
enum file_encryption {
	ENCRYPTED,
	UNENCRYPTED
};


// Structure which contains an abstraction for an open file.
typedef struct {
	// The data of the file in 128-bit blocks.
	block128* blocks;
	// The name of the file.
	char* name;
	// The number of blocks.
	int block_count;
	// The size of the file in bytes.
	long long size;
	// The number of bytes padded onto the end of the file.
	int padding;
	// File descriptor for Linux file I/O.
	int fd;
	// Whether the file is encrypted or unencrypted.
	enum file_encryption flag;
} file_t;


/**	Initialize the specified file_t. This opens the file and performs any other
 *	low-level details.
 *	@return	NULL on success; exception_t* on failure.
 */
exception_t* file_init(const char* file_name, enum file_encryption flag, file_t* file);


/**	Frees the specified file_t. This closes the file and performs any other
 *	low-level details.
 *	@return	NULL on success; exception_t* on failure.
 */
exception_t* file_free(file_t* file);


/**	Gets the total number of block128s that exist within the file.
 *	@out	block_count: The number of blocks in the file.
 *	@return	NULL on success; exception_t* on failure.
 */
exception_t* file_get_block_count(file_t* file, long long* block_count);


/**	Read in the file data and metadata as per the file's encrypted/decrypted status.
 *	@param	block_index: The index of the first block to read in.
 *	@param	block_count: The number of 128-bit blocks to read in.
 *	@out	blocks: Pointer to the array of blocks read in.
 *	@out	blocks_read: The number of blocks read in.
 *	@return	NULL on success; exception_t* on failure.
 */
exception_t* file_read(file_t* file, int block_index, int block_count, block128** blocks, int* blocks_read);


/**	Private function which reads in file metadata.
 *	@return	NULL on success; exception_t* on failure.
 */
exception_t* file_read_metadata(file_t* file);


/**	Write the data and metadata based on the file's encrypted/decrypted status.
 *	@param	block_index: The index of the first block to write to.
 *	@param	block_count: The number of blocks to write.
 *	@param	blocks: Pointer to the array of blocks to write.
 *	@out	blocks_written: The number of blocks written.
 *	@return	NULL on success; exception_t* on failure.
 */
exception_t* file_write(file_t* file, int block_index, int block_count, block128* blocks, int* blocks_written);


/**	Private function which writes file metadata.
 *	@return NULL on success; exception_t* on failure.
 */
exception_t* file_write_metadata(file_t* file);


/**	Private function to write file padding into an unencrypted file.
 *	@return NULL on success; exception_t* on failure.
 */
exception_t* file_write_padding(file_t* file);


#endif //file_H
