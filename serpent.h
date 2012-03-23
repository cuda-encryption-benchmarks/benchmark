#ifndef serpent_H
#define serpent_H


#include "block128.h"
#include "ccc/ccc.h"
#include "mirror_bytes.h"
#include "typedef.h"


// Rotate the bits in the specified number x left by the specified number n.
#define rotl_fixed(x, n)   (((x) << (n)) | ((x) >> (32 - (n))))
// Rotate the bits in the specified number x right by the specified number n.
#define rotr_fixed(x, n)   (((x) >> (n)) | ((x) << (32 - (n))))
// Arbitrary amount of memory to subtract from total free memory.
#define SERPENT_CUDA_MEMORY_BUFFER 500000


// Structure to hold key for serpent algorithm.
typedef struct {
	// The first part of the key.
	block128 key0;
	// The second part of the key.
	block128 key1;
} serpent_key;

/**	Run the specified array of 128-but blocks through the Serpent encryption algorithm.
 *	@return NULL on success, exception_t* on failure.
 */
exception_t* serpent(serpent_key* user_key, block128* blocks, int block_count, enum mode mode, enum encryption encryption);


/**	Decrypt the specified array of 128-bit blocks through the CUDA Serpent algorithm.
 *	@return	NULL on success, exception_t* on failure.
 */
exception_t* serpent_cuda_decrypt(serpent_key* user_key, block128* blocks, int block_count);


/**	Encrypt the specified array of 128-bit blocks through the CUDA Serpent algorithm.
 *	@return	NULL on success, exception_t* on failure.
 */
exception_t* serpent_cuda_encrypt(serpent_key* user_key, block128* blocks, int block_count);

/**	Private inner function to prevent linking errors because nvcc does not like external libraries.
 *	@return 0 on success, -1 on failure.
 */
#ifdef __cplusplus
extern "C"
#endif
int serpent_cuda_decrypt_cu(uint32* subkey, block128* blocks, int block_count);

/**	Private inner function to prevent linking errors because nvcc does not like external libraries.
 *	@return 0 on success, -1 on failure.
 */
#ifdef __cplusplus
extern "C"
#endif
int serpent_cuda_encrypt_cu(uint32* subkey, block128* blocks, int block_count);


/**	Decrypt the specified array of 128-bit blocks in parallel through the Serpent encryption algorithm.
 *	@return NULL on success, exception_t* on failure.
 */
exception_t* serpent_parallel_decrypt(serpent_key* user_key, block128* blocks, int block_count);


/**	Encrypt the specified array of 128-bit blocks in parallel through the Serpent encryption algorithm.
 *	@return NULL on success, exception_t* on failure.
 */
exception_t* serpent_parallel_encrypt(serpent_key* user_key, block128* blocks, int block_count);


/**	Decrypt the specified array of 128-bit blocks serially through the Serpent encryption algorithm.
 *	@return NULL on success, exception_t* on failure.
 */
exception_t* serpent_serial_decrypt(serpent_key* user_key, block128* blocks, int block_count);


/**	Encrypt the specified array of 128-bit blocks serially through the Serpent encryption algorithm.
 *	@return	NULL on success, exception_t* on failure.
 */
exception_t* serpent_serial_encrypt(serpent_key* user_key, block128* blocks, int block_count);


/**	Private function that generates the subkey for the Serpent encryption algorithm.
 *	@out	subkey: Pointer to an array of uint32 that represent the subkeys for the serpent algorithm.
 *	@return NULL on success, exception_t* on failure.
 */
exception_t* serpent_init_subkey(serpent_key* user_key, uint32** subkey);


/**	Private function that serves as a reminder to free the subkey.
 *	@return	NULL on success, exception_t* on failure.
 */
exception_t* serpent_free_subkey(uint32* subkey);


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
