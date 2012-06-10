// serpent.cpp - written and placed in the public domain by Wei Dai.

/* Subsequently adapted for TrueCrypt (copyrighted by its respectful owners)... 
   and then subsequently re-adapted for CUDA benchmarking. */

// A copy of the TrueCrypt license is contained in the LICENSE file.


#ifndef serpent_H
#define serpent_H

#ifdef DEBUG
#define DEBUG_SERPENT
#endif


#include <inttypes.h>
#include <omp.h>
#include <stdio.h>

#include "block128.h"
#include "ccc/ccc.h"
#include "cuda_extension.h"
#include "mirror_bytes.h"
#include "key.h"
#include "typedef.h"


//! Rotate the bits in the specified number x left by the specified number n.
#define rotl_fixed(x, n)   (((x) << (n)) | ((x) >> (32 - (n))))
//! Rotate the bits in the specified number x right by the specified number n.
#define rotr_fixed(x, n)   (((x) >> (n)) | ((x) << (32 - (n))))

/*! Percentage of global memory to subtract per iteration when trying to allocate
 *  a memory buffer for blocks in CUDA.
 */
#define SERPENT_CUDA_MEMORY_MULTIPLIER 0.001


/*!	\brief Run the specified array of 128-bit blocks through the Serpent encryption or decryption algorithm in the specified mode.
 * 	
 * 	\param[in]	user_key	The user-supplied key.
 * 	\param[in]	blocks		The blocks to run through the Serpent algorithm.
 * 	\param[in]	block_count	The number of blocks in blocks.
 * 	\param[in]	mode		How to run the Serpent algorithm.
 * 	\param[in]	encryption	Whether to encrypt or decrypt the blocks.
 *	\param[out]	buffer_size	Size of the global memory buffer used (only for CUDA).
 *	\return NULL on success, exception_t* on failure.
 */
exception_t* serpent(key256_t* user_key, block128_t* blocks, int block_count, enum mode mode, enum encryption encryption, size_t* buffer_size);


/*!	\brief Decrypt the specified array of 128-bit blocks through the CUDA Serpent algorithm.
 *	\return	NULL on success, exception_t* on failure.
 */
exception_t* serpent_cuda_decrypt(key256_t* user_key, block128_t* blocks, int block_count, size_t* buffer_size);


/*!	\brief Encrypt the specified array of 128-bit blocks through the CUDA Serpent algorithm.
 *	@return	NULL on success, exception_t* on failure.
 */
exception_t* serpent_cuda_encrypt(key256_t* user_key, block128_t* blocks, int block_count, size_t* buffer_size);


/*!	\brief Private inner function to prevent linking errors with the external library.
 *
 *	\return 0 on success, -1 on failure.
 */
#ifdef __cplusplus
extern "C"
#endif
int serpent_cuda_decrypt_cu(uint32_t* subkey, block128_t* blocks, int block_count, size_t* buffer_size);


/*!	\brief Private inner function to prevent linking errors with the external library.
 *
 *	\return 0 on success, -1 on failure.
 */
#ifdef __cplusplus
extern "C"
#endif
int serpent_cuda_encrypt_cu(uint32_t* subkey, block128_t* blocks, int block_count, size_t* buffer_size);


/*!	\brief Private function that allocates a memory buffer for blocks on the GPU. 
 * 
 * 	\param[in]	free_global_memory		The maximum amount of free global memory on the graphics card (may still be fragmented).
 * 	\param[in]	total_global_memory		The total amount of global memory on the graphics card.
 * 	\param[in]	block_count			The number of blocks to be allocated.
 * 	\param[in]	multiprocessor_count		The number of multiprocessors to be used.
 * 	\param[in]	thread_count			The number of threads to be used.
 *	\param[out]	cuda_blocks			Pointer to the memory allocated for the blocks on the GPU.
 *	\param[out]	used_global_memory		Amount of memory used on the GPU.
 *	\param[out]	blocks_per_kernel		Number of block128_t each kernel can modify.
 *	\param[out]	blocks_per_thread		Number of block128_t each thread will modify.
 *	\param[out]	buffer_allocation_attempts	Number of attempts made to allocate memory on the GPU.
 *	\return	0 on success, -1 on failure.
 */
#ifdef __cplusplus
extern "C"
#endif
int serpent_cuda_allocate_buffer(size_t free_global_memory, size_t total_global_memory, int block_count, int multiprocessor_count, int thread_count, block128_t** cuda_blocks, size_t* used_global_memory, int* blocks_per_kernel, int* blocks_per_thread, int* buffer_allocation_attempts);


/*!	\brief Decrypt the specified array of 128-bit blocks in parallel through the Serpent encryption algorithm.
 * 
 *	\return NULL on success, exception_t* on failure.
 */
exception_t* serpent_parallel_decrypt(key256_t* user_key, block128_t* blocks, int block_count);


/*!	\brief Private function to decrypt a single block of serpent.
 */
void serpent_decrypt_block(block128_t* block, uint32_t* subkey);


/*!	\brief Private function to encrypt a single block of serpent.
 */
void serpent_encrypt_block(block128_t* block, uint32_t* subkey);


/*!	\brief Encrypt the specified array of 128-bit blocks in parallel through the Serpent encryption algorithm.
 * 
 *	\return NULL on success, exception_t* on failure.
 */
exception_t* serpent_parallel_encrypt(key256_t* user_key, block128_t* blocks, int block_count);


/*!	\brief Decrypt the specified array of 128-bit blocks serially through the Serpent encryption algorithm.
 * 
 *	\return NULL on success, exception_t* on failure.
 */
exception_t* serpent_serial_decrypt(key256_t* user_key, block128_t* blocks, int block_count);


/*!	\brief Encrypt the specified array of 128-bit blocks serially through the Serpent encryption algorithm.
 *
 *	\return	NULL on success, exception_t* on failure.
 */
exception_t* serpent_serial_encrypt(key256_t* user_key, block128_t* blocks, int block_count);


/*!	\brief Private function that generates the subkey for the Serpent encryption algorithm.
 *
 * 	\param[in]	user_key	The user-supplied key.
 *	\param[out]	subkey		Pointer to an array of uint32_t that represent the subkeys for the serpent algorithm.
 *	\return NULL on success, exception_t* on failure.
 */
exception_t* serpent_init_subkey(key256_t* user_key, uint32_t** subkey);


/*!	\brief Private function that serves as a reminder to free the subkey.
 *
 *	\return	NULL on success, exception_t* on failure.
 */
exception_t* serpent_free_subkey(uint32_t* subkey);


#endif // serpent_H
