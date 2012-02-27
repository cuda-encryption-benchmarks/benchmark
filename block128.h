#ifndef block128_H
#define block128_H

#include <stdint.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#include "ccc/ccc.h"


typedef uint32_t uint32;


// Structure to hold a 128-bit binary stream.
typedef struct {
	// The first 32 btis of data.
	uint32 x0;
	// The second 32 bits of data.
	uint32 x1;
	// The third 32 bits of data.
	uint32 x2;
	// The fourth 32 bits of data.
	uint32 x3;
} block128;


/**	Read in a 128-bit block from the specified file descriptor.
 *	@out	block: The 128-bit block read in from the file.
 *	@return	NULL on success, exception_t* on failure.
 */
exception_t* block128_read_block(int fd, block128* block);


/**	Read in an array of 128-bit blocks from the specified file descriptor.
 *	@out	blocks: The array of 128-bit blocks read in from the file.
 *	@out	block_count: The number of 128-bit blocks read in from the file.
 *	@return	NULL on success, exception_t* on failure. 
 */
exception_t* block128_read_file(int fd, block128** blocks, int* block_count);


/**	Write a 128-bit block to the specified file descriptor.
 *	@return	NULL on success, exception_t* on failure.
 */
exception_t* block128_write_block(int fd, block128* block);


/**	Write the specified array of 128-bit blocks to the specified file descriptor.
 *	@return NULL on success, exception_t* on failure.
 */
exception_t* block128_write_file(int fd, block128* blocks, int block_count);


#endif
