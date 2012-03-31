#ifndef key_H
#define key_H


#include "block128.h"
#include "ccc/ccc.h"


// A structure which allows for an abstraction of a 256-bit key.
typedef struct {
	// The first part of the key.
	block128 key0;
	// The second part of the key.
	block128 key1;
} key256_t;


/**     Get the word of the specified key at the specified index.
 *      @out    word: A uint32 representing the key at the specified index.
 *      @return NULL on success, exception_t* on failure.
 */
exception_t* key256_get_word(key256_t* key, int index, uint32* word);


/**     Set the word of the specified key at the specified index to the specified value.
 *      @return NULL on success, exception_t* on failure.
 */
exception_t* key256_set_word(key256_t* key, int index, uint32 word);


#endif
