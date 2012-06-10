#ifndef key_H
#define key_H


#include "block128.h"
#include "ccc/ccc.h"


/*!	\brief A structure which is an abstraction of a user-supplied 256-bit key.
 */
typedef struct {
	//! The first 128 bits of the key.
	block128_t key0;
	//! The second 128 bits of the key.
	block128_t key1;
} key256_t;


/*!	\brief Returns the 32-bit word of the specified key at the specified index.
 *
 * 	\param[in]	key	The key to get the word from.
 * 	\param[in]	index	The word of the key to get.
 *	\param[out]	word	A uint32_t representing the key at the specified index.
 *	\return NULL on success, exception_t* on failure.
 */
exception_t* key256_get_word(key256_t* key, int index, uint32_t* word);


/*!	\brief Set the word of the specified key at the specified index to the specified value.
 *	\return NULL on success, exception_t* on failure.
 */
exception_t* key256_set_word(key256_t* key, int index, uint32_t word);


#endif
