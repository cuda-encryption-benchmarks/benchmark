#ifndef serpent_H
#define serpent_H


#include "block128.h"
#include "ccc/ccc.h"
#include "mirror_bytes.h"


// Rotate the bits in the specified number x left by the specified number n.
#define rotlFixed(x, n)   (((x) << (n)) | ((x) >> (32 - (n))))
// Rotate the bits in the specified number x right by the specified number n.
#define rotrFixed(x, n)   (((x) >> (n)) | ((x) << (32 - (n))))


// Structure to hold key for serpent algorithm.
typedef struct {
	// The first part of the key.
	block128 key0;
	// The second part of the key.
	block128 key1;
} serpent_key;


/**	Encrypt the specified array of 128-bit blocks serially through the Serpent encryption algorithm.
 *	@return	NULL on success, exception_t* on failure.
 */
exception_t* serpent_encrypt_serial(serpent_key* user_key, block128* blocks, int block_count);


/**	Private function that generates the subkey for the Serpent encryption algorithm.
 *	@out	subkey: Pointer to an array of uint32 that represent the subkeys for the serpent algorithm.
 *	@return NULL on success, exception_t* on failure.
 */
exception_t* serpent_init_key(serpent_key* user_key, uint32** subkey);


/**	Private function that serves as a reminder to free the subkey.
 *	@return	NULL on success, exception_t* on failure.
 */
exception_t* serpent_free_key(serpent_key* user_key, uint32* subkey);


/**	Get the word of the specified key at the specified index.
 *	@out	word: A uint32 representing the key at the specified index.
 *	@return	NULL on success, exception_t* on failure.
 */
exception_t* serpent_key_get_word(serpent_key* key, int index, uint32* word);


/**	Set the word of the specified key at the specified index to the specified value.
 *	@return	NULL on success, exception_t* on failure.
 */
exception_t* serpent_key_set_word(serpent_key* key, int index, uint32 word);


#endif // serpent_H
