#ifndef serpent_H
#define serpent_H


#include "block128.h"
#include "ccc/ccc.h"


/**	Encrypt the specified array of 128-bit blocks serially through the Serpent encryption algorithm.
 *	@return	NULL on success, exception_t* on failure.
 */
exception_t* serpent_encrypt_serial(block128* user_key, block128* blocks, int block_count);


/**	Serially generate the key for the Serpent encryption algorithm.
 *	@return NULL on success, exception_t* on failure.
 */
exception_t* serpent_generate_key_serial(block128* user_key, block128** serpent_key);


#endif // serpent_H
