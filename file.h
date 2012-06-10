#ifndef file_H
#define file_H

//! We have no idea why this works but it prevent an 'implicit declaration of ftruncate()'.
#define _BSD_SOURCE
//! Feature test macro for opening large files.
#define _LARGEFILE64_SOURCE

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#include "block128.h"
#include "ccc/ccc.h"


/*! \brief Enumeration which specifies whether the file to be read in is encrypted or not.
 *  \note This should not be confused with the encryption enumeration. This should be
 *  	deprecated eventually.
 */
enum file_encryption {
	ENCRYPTED,
	UNENCRYPTED
};


/*! \brief Structure which contains an abstraction for a file that is to be encrypted
 * 	or decrypted.
 */
typedef struct {
	//! The data of the file in 128-bit blocks.
	block128_t* blocks;
	//! The name of the file.
	char* name;
	//! The number of blocks.
	int block_count;
	//! The size of the file in bytes.
	long long size;
	//! The number of bytes padded onto the end of the file.
	long long padding;
	//! File descriptor for Linux file I/O.
	int fd;
	//! Whether the file is encrypted or unencrypted.
	enum file_encryption flag;
} file_t;


/*!	\brief Initialize the specified file_t. This opens the file and performs any other
 *		low-level details, such as adding or removing metadata.
 *	\warning Files that are not properly closed will become corrupt and may lose data.
 *
 *	\return	NULL on success; exception_t* on failure.
 */
exception_t* file_init(const char* file_name, enum file_encryption flag, file_t* file);


/*!	\brief Frees the specified file_t. This closes the file and performs any other
 *		low-level details such as adding or removing metadata.
 *
 *	\return	NULL on success; exception_t* on failure.
 */
exception_t* file_free(file_t* file);


/*!	\brief Gets the total number of 128-bit blocks that exist within the file.
 *
 * 	\param[in]	file		The file to get the number of 128-bit blocks from.
 *	\param[out]	block_count	The number of blocks in the file.
 *	\return	NULL on success; exception_t* on failure.
 */
exception_t* file_get_block_count(file_t* file, long long* block_count);


/*!	\brief Read in the file data and metadata as per the file's encrypted/decrypted status. 
 * 	\warning This will automatically allocate space for the amount of data read in. This data
 * 		should later be freed by calling file_write().
 *
 * 	\param[in]	file		The files whose data is to be read.
 *	\param[in]	block_index	The index of the first block to read in.
 *	\param[in]	block_count	The number of 128-bit blocks to read in.
 *	\param[out]	blocks		Pointer to the array of blocks read in.
 *	\return	NULL on success; exception_t* on failure.
 */
exception_t* file_read(file_t* file, int block_index, int block_count, block128_t** blocks);


/*!	\brief Private function which reads in file metadata.
 *
 *	\return	NULL on success; exception_t* on failure.
 */
exception_t* file_read_metadata(file_t* file);


/*!	\brief Write the data and metadata based on the file's encrypted/decrypted status.
 * 	\warning This will automatically free the data pointed to by blocks after it is written.
 *
 * 	\param[in]	file		The file whose data is to be written.
 *	\param[in]	block_index	The index of the first block to write to.
 *	\param[in]	block_count	The number of blocks to write.
 *	\param[in]	blocks		Pointer to the array of blocks to write.
 *	\return	NULL on success; exception_t* on failure.
 */
exception_t* file_write(file_t* file, int block_index, int block_count, block128_t* blocks);


/*!	\brief Private function which writes file metadata.
 *
 *	\return NULL on success; exception_t* on failure.
 */
exception_t* file_write_metadata(file_t* file);


/*!	\brief Private function to write file padding into an unencrypted file.
 *
 * 	\return NULL on success; exception_t* on failure.
 */
exception_t* file_write_padding(file_t* file);


#endif //file_H
