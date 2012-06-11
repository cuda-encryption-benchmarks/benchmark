#ifndef block128_H
#define block128_H

#include <stdint.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#include "ccc/ccc.h"


//! The number of bytes in single 128-bit block. Used for clarity purposes.
#define BYTES_PER_BLOCK128 16


/*! \brief Abstraction for holding a 128-bit binary stream.
 */
typedef struct {
	//! The first 32 bits of data.
	uint32_t x0;
	//! The second 32 bits of data.
	uint32_t x1;
	//! The third 32 bits of data.
	uint32_t x2;
	//! The fourth 32 bits of data.
	uint32_t x3;
} block128_t;


/*!	\brief Read in a 128-bit block from the specified file descriptor.
 * 	\deprecated Extremely slow; poorly-implemented. Use file_read() instead.
 *
 *	\param[in]	fd	The file to read the block from.
 *	\param[out]	block	The 128-bit block read in from the file.
 *	\return	NULL on success, exception_t* on failure.
 */
exception_t* block128_read(int fd, block128_t* block);


/*!	\brief Write a 128-bit block to the specified file descriptor.
 * 	\deprecated Extremely slow; poorly-implemented. Use file_write() instead.
 * 
 *	\return	NULL on success, exception_t* on failure.
 */
exception_t* block128_write(int fd, block128_t* block);


#endif
