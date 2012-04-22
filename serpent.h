// serpent.cpp - written and placed in the public domain by Wei Dai.

/* Subsequently adapted for TrueCrypt (copyrighted by its respectful owners)... 
   and then subsequently re-adapted for CUDA benchmarking. */

// A copy of the TrueCrypt license is contained in the LICENSE file.


#ifndef serpent_H
#define serpent_H


#include <inttypes.h>
#include <omp.h>
#include <stdio.h>

#include "block128.h"
#include "ccc/ccc.h"
#include "cuda_extension.h"
#include "mirror_bytes.h"
#include "key.h"
#include "typedef.h"


// Rotate the bits in the specified number x left by the specified number n.
#define rotl_fixed(x, n)   (((x) << (n)) | ((x) >> (32 - (n))))
// Rotate the bits in the specified number x right by the specified number n.
#define rotr_fixed(x, n)   (((x) >> (n)) | ((x) << (32 - (n))))
// Percentage of global memory to subtract per iteration when trying to allocate
// a memory buffer for blocks in CUDA.
#define SERPENT_CUDA_MEMORY_MULTIPLIER 0.001


/**	Run the specified array of 128-bit blocks through the Serpent encryption algorithm.
 *	@out	buffer_size: Size of the global memory buffer used (only for CUDA).
 *	@return NULL on success, exception_t* on failure.
 */
exception_t* serpent(key256_t* user_key, block128_t* blocks, int block_count, enum mode mode, enum encryption encryption, size_t* buffer_size);


/**	Decrypt the specified array of 128-bit blocks through the CUDA Serpent algorithm.
 *	@return	NULL on success, exception_t* on failure.
 */
exception_t* serpent_cuda_decrypt(key256_t* user_key, block128_t* blocks, int block_count, size_t* buffer_size);


/**	Encrypt the specified array of 128-bit blocks through the CUDA Serpent algorithm.
 *	@return	NULL on success, exception_t* on failure.
 */
exception_t* serpent_cuda_encrypt(key256_t* user_key, block128_t* blocks, int block_count, size_t* buffer_size);


/**	Private inner function to prevent linking errors because nvcc does not like external libraries.
 *	@return 0 on success, -1 on failure.
 */
#ifdef __cplusplus
extern "C"
#endif
int serpent_cuda_decrypt_cu(uint32_t* subkey, block128_t* blocks, int block_count, size_t* buffer_size);


/**	Private inner function to prevent linking errors because nvcc does not like external libraries.
 *	@return 0 on success, -1 on failure.
 */
#ifdef __cplusplus
extern "C"
#endif
int serpent_cuda_encrypt_cu(uint32_t* subkey, block128_t* blocks, int block_count, size_t* buffer_size);


/**	Private function that allocates the buffer for blocks on the GPU. 
 *	@out	cuda_blocks: Pointer to the memory allocated for the blocks on the GPU.
 *	@out	used_global_memory: Amount of memory used on the GPU.
 *	@out	blocks_per_kernel: Number of block128_t each kernel can modify.
 *	@out	blocks_per_thread: Number of block128_t each thread will modify.
 *	@out	buffer_allocation_attempts: Number of attempts made to allocate memory on the GPU.
 *	@return	0 on success, -1 on failure.
 */
#ifdef __cplusplus
extern "C"
#endif
int serpent_cuda_allocate_buffer(size_t free_global_memory, size_t total_global_memory, int block_count, int multiprocessor_count, int thread_count, block128_t** cuda_blocks, size_t* used_global_memory, int* blocks_per_kernel, int* blocks_per_thread, int* buffer_allocation_attempts);


/**	Decrypt the specified array of 128-bit blocks in parallel through the Serpent encryption algorithm.
 *	@return NULL on success, exception_t* on failure.
 */
exception_t* serpent_parallel_decrypt(key256_t* user_key, block128_t* blocks, int block_count);


/**	Private function to decrypt a single block of serpent.
 *	@return	NULL on success, exception_t* on failure. 
 */
exception_t* serpent_decrypt_block(block128_t* block, uint32_t* subkey);


/**	Private function to encrypt a single block of serpent.
 *	@return	NULL on success, exception_t* on failure.
 */
exception_t* serpent_encrypt_block(block128_t* block, uint32_t* subkey);


/**	Encrypt the specified array of 128-bit blocks in parallel through the Serpent encryption algorithm.
 *	@return NULL on success, exception_t* on failure.
 */
exception_t* serpent_parallel_encrypt(key256_t* user_key, block128_t* blocks, int block_count);


/**	Decrypt the specified array of 128-bit blocks serially through the Serpent encryption algorithm.
 *	@return NULL on success, exception_t* on failure.
 */
exception_t* serpent_serial_decrypt(key256_t* user_key, block128_t* blocks, int block_count);


/**	Encrypt the specified array of 128-bit blocks serially through the Serpent encryption algorithm.
 *	@return	NULL on success, exception_t* on failure.
 */
exception_t* serpent_serial_encrypt(key256_t* user_key, block128_t* blocks, int block_count);


/**	Private function that generates the subkey for the Serpent encryption algorithm.
 *	@out	subkey: Pointer to an array of uint32_t that represent the subkeys for the serpent algorithm.
 *	@return NULL on success, exception_t* on failure.
 */
exception_t* serpent_init_subkey(key256_t* user_key, uint32_t** subkey);


/**	Private function that serves as a reminder to free the subkey.
 *	@return	NULL on success, exception_t* on failure.
 */
exception_t* serpent_free_subkey(uint32_t* subkey);


#endif // serpent_H
