#ifndef block128_H
#define block128_H

#include <stdint.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#include "ccc/ccc.h"


// Structure to hold a 128-bit binary stream.
typedef struct {
	// The first 32 btis of data.
	uint32_t x0;
	// The second 32 bits of data.
	uint32_t x1;
	// The third 32 bits of data.
	uint32_t x2;
	// The fourth 32 bits of data.
	uint32_t x3;
} block128_t;


/**	DEPRECATED. Read in a 128-bit block from the specified file descriptor.
 *	@out	block: The 128-bit block read in from the file.
 *	@return	NULL on success, exception_t* on failure.
 */
exception_t* block128_read(int fd, block128_t* block);


/**	DEPRECATED. Write a 128-bit block to the specified file descriptor.
 *	@return	NULL on success, exception_t* on failure.
 */
exception_t* block128_write(int fd, block128_t* block);


#endif
