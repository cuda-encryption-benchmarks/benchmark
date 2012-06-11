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

// order of output from S-box functions
#define beforeS0(f) f(0,a,b,c,d,e)
#define afterS0(f) f(1,b,e,c,a,d)
#define afterS1(f) f(2,c,b,a,e,d)
#define afterS2(f) f(3,a,e,b,d,c)
#define afterS3(f) f(4,e,b,d,c,a)
#define afterS4(f) f(5,b,a,e,c,d)
#define afterS5(f) f(6,a,c,b,e,d)
#define afterS6(f) f(7,a,c,d,b,e)
#define afterS7(f) f(8,d,e,b,a,c)

// order of output from inverse S-box functions
#define beforeI7(f) f(8,a,b,c,d,e)
#define afterI7(f) f(7,d,a,b,e,c)
#define afterI6(f) f(6,a,b,c,e,d)
#define afterI5(f) f(5,b,d,e,c,a)
#define afterI4(f) f(4,b,c,e,a,d)
#define afterI3(f) f(3,a,b,e,c,d)
#define afterI2(f) f(2,b,d,e,c,a)
#define afterI1(f) f(1,a,b,c,e,d)
#define afterI0(f) f(0,a,d,b,e,c)

// The linear transformation.
#define linear_transformation(i,a,b,c,d,e) {\
        a = rotl_fixed(a, 13);   \
        c = rotl_fixed(c, 3);    \
        d = rotl_fixed(d ^ c ^ (a << 3), 7);     \
        b = rotl_fixed(b ^ a ^ c, 1);    \
        a = rotl_fixed(a ^ b ^ d, 5);       \
        c = rotl_fixed(c ^ d ^ (b << 7), 22);}

// The inverse linear transformation.
#define inverse_linear_transformation(i,a,b,c,d,e)        {\
        c = rotr_fixed(c, 22);   \
        a = rotr_fixed(a, 5);    \
        c ^= d ^ (b << 7);      \
        a ^= b ^ d;             \
        b = rotr_fixed(b, 1);    \
        d = rotr_fixed(d, 7) ^ c ^ (a << 3);     \
        b ^= a ^ c;             \
        c = rotr_fixed(c, 3);    \
        a = rotr_fixed(a, 13);}

// The instruction sequences for the S-box functions
// come from Dag Arne Osvik's paper "Speeding Up Serpent".
#define S0(i, r0, r1, r2, r3, r4) \
       {           \
    r3 ^= r0;   \
    r4 = r1;   \
    r1 &= r3;   \
    r4 ^= r2;   \
    r1 ^= r0;   \
    r0 |= r3;   \
    r0 ^= r4;   \
    r4 ^= r3;   \
    r3 ^= r2;   \
    r2 |= r1;   \
    r2 ^= r4;   \
    r4 = ~r4;      \
    r4 |= r1;   \
    r1 ^= r3;   \
    r1 ^= r4;   \
    r3 |= r0;   \
    r1 ^= r3;   \
    r4 ^= r3;   \
            }

#define I0(i, r0, r1, r2, r3, r4) \
       {           \
    r2 = ~r2;      \
    r4 = r1;   \
    r1 |= r0;   \
    r4 = ~r4;      \
    r1 ^= r2;   \
    r2 |= r4;   \
    r1 ^= r3;   \
    r0 ^= r4;   \
    r2 ^= r0;   \
    r0 &= r3;   \
    r4 ^= r0;   \
    r0 |= r1;   \
    r0 ^= r2;   \
    r3 ^= r4;   \
    r2 ^= r1;   \
    r3 ^= r0;   \
    r3 ^= r1;   \
    r2 &= r3;   \
    r4 ^= r2;   \
            }

#define S1(i, r0, r1, r2, r3, r4) \
       {           \
    r0 = ~r0;      \
    r2 = ~r2;      \
    r4 = r0;   \
    r0 &= r1;   \
    r2 ^= r0;   \
    r0 |= r3;   \
    r3 ^= r2;   \
    r1 ^= r0;   \
    r0 ^= r4;   \
    r4 |= r1;   \
    r1 ^= r3;   \
    r2 |= r0;   \
    r2 &= r4;   \
    r0 ^= r1;   \
    r1 &= r2;   \
    r1 ^= r0;   \
    r0 &= r2;   \
    r0 ^= r4;   \
            }

#define I1(i, r0, r1, r2, r3, r4) \
       {           \
    r4 = r1;   \
    r1 ^= r3;   \
    r3 &= r1;   \
    r4 ^= r2;   \
    r3 ^= r0;   \
    r0 |= r1;   \
    r2 ^= r3;   \
    r0 ^= r4;   \
    r0 |= r2;   \
    r1 ^= r3;   \
    r0 ^= r1;   \
    r1 |= r3;   \
    r1 ^= r0;   \
    r4 = ~r4;      \
    r4 ^= r1;   \
    r1 |= r0;   \
    r1 ^= r0;   \
    r1 |= r4;   \
    r3 ^= r1;   \
            }

#define S2(i, r0, r1, r2, r3, r4) \
       {           \
    r4 = r0;   \
    r0 &= r2;   \
    r0 ^= r3;   \
    r2 ^= r1;   \
    r2 ^= r0;   \
    r3 |= r4;   \
    r3 ^= r1;   \
    r4 ^= r2;   \
    r1 = r3;   \
    r3 |= r4;   \
    r3 ^= r0;   \
    r0 &= r1;   \
    r4 ^= r0;   \
    r1 ^= r3;   \
    r1 ^= r4;   \
    r4 = ~r4;      \
            }

#define I2(i, r0, r1, r2, r3, r4) \
       {           \
    r2 ^= r3;   \
    r3 ^= r0;   \
    r4 = r3;   \
    r3 &= r2;   \
    r3 ^= r1;   \
    r1 |= r2;   \
    r1 ^= r4;   \
    r4 &= r3;   \
    r2 ^= r3;   \
    r4 &= r0;   \
    r4 ^= r2;   \
    r2 &= r1;   \
    r2 |= r0;   \
    r3 = ~r3;      \
    r2 ^= r3;   \
    r0 ^= r3;   \
    r0 &= r1;   \
    r3 ^= r4;   \
    r3 ^= r0;   \
            }

#define S3(i, r0, r1, r2, r3, r4) \
       {           \
    r4 = r0;   \
    r0 |= r3;   \
    r3 ^= r1;   \
    r1 &= r4;   \
    r4 ^= r2;   \
    r2 ^= r3;   \
    r3 &= r0;   \
    r4 |= r1;   \
    r3 ^= r4;   \
    r0 ^= r1;   \
    r4 &= r0;   \
    r1 ^= r3;   \
    r4 ^= r2;   \
    r1 |= r0;   \
    r1 ^= r2;   \
    r0 ^= r3;   \
    r2 = r1;   \
    r1 |= r3;   \
    r1 ^= r0;   \
            }

#define I3(i, r0, r1, r2, r3, r4) \
       {           \
    r4 = r2;   \
    r2 ^= r1;   \
    r1 &= r2;   \
    r1 ^= r0;   \
    r0 &= r4;   \
    r4 ^= r3;   \
    r3 |= r1;   \
    r3 ^= r2;   \
    r0 ^= r4;   \
    r2 ^= r0;   \
    r0 |= r3;   \
    r0 ^= r1;   \
    r4 ^= r2;   \
    r2 &= r3;   \
    r1 |= r3;   \
    r1 ^= r2;   \
    r4 ^= r0;   \
    r2 ^= r4;   \
            }

#define S4(i, r0, r1, r2, r3, r4) \
       {           \
    r1 ^= r3;   \
    r3 = ~r3;      \
    r2 ^= r3;   \
    r3 ^= r0;   \
    r4 = r1;   \
    r1 &= r3;   \
    r1 ^= r2;   \
    r4 ^= r3;   \
    r0 ^= r4;   \
    r2 &= r4;   \
    r2 ^= r0;   \
    r0 &= r1;   \
    r3 ^= r0;   \
    r4 |= r1;   \
    r4 ^= r0;   \
    r0 |= r3;   \
    r0 ^= r2;   \
    r2 &= r3;   \
    r0 = ~r0;      \
    r4 ^= r2;   \
            }

#define I4(i, r0, r1, r2, r3, r4) \
       {           \
    r4 = r2;   \
    r2 &= r3;   \
    r2 ^= r1;   \
    r1 |= r3;   \
    r1 &= r0;   \
    r4 ^= r2;   \
    r4 ^= r1;   \
    r1 &= r2;   \
    r0 = ~r0;      \
    r3 ^= r4;   \
    r1 ^= r3;   \
    r3 &= r0;   \
    r3 ^= r2;   \
    r0 ^= r1;   \
    r2 &= r0;   \
    r3 ^= r0;   \
    r2 ^= r4;   \
    r2 |= r3;   \
    r3 ^= r0;   \
    r2 ^= r1;   \
            }

#define S5(i, r0, r1, r2, r3, r4) \
       {           \
    r0 ^= r1;   \
    r1 ^= r3;   \
    r3 = ~r3;      \
    r4 = r1;   \
    r1 &= r0;   \
    r2 ^= r3;   \
    r1 ^= r2;   \
    r2 |= r4;   \
    r4 ^= r3;   \
    r3 &= r1;   \
    r3 ^= r0;   \
    r4 ^= r1;   \
    r4 ^= r2;   \
    r2 ^= r0;   \
    r0 &= r3;   \
    r2 = ~r2;      \
    r0 ^= r4;   \
    r4 |= r3;   \
    r2 ^= r4;   \
            }

#define I5(i, r0, r1, r2, r3, r4) \
       {           \
    r1 = ~r1;      \
    r4 = r3;   \
    r2 ^= r1;   \
    r3 |= r0;   \
    r3 ^= r2;   \
    r2 |= r1;   \
    r2 &= r0;   \
    r4 ^= r3;   \
    r2 ^= r4;   \
    r4 |= r0;   \
    r4 ^= r1;   \
    r1 &= r2;   \
    r1 ^= r3;   \
    r4 ^= r2;   \
    r3 &= r4;   \
    r4 ^= r1;   \
    r3 ^= r0;   \
    r3 ^= r4;   \
    r4 = ~r4;      \
            }

#define S6(i, r0, r1, r2, r3, r4) \
       {           \
    r2 = ~r2;      \
    r4 = r3;   \
    r3 &= r0;   \
    r0 ^= r4;   \
    r3 ^= r2;   \
    r2 |= r4;   \
    r1 ^= r3;   \
    r2 ^= r0;   \
    r0 |= r1;   \
    r2 ^= r1;   \
    r4 ^= r0;   \
    r0 |= r3;   \
    r0 ^= r2;   \
    r4 ^= r3;   \
    r4 ^= r0;   \
    r3 = ~r3;      \
    r2 &= r4;   \
    r2 ^= r3;   \
            }

#define I6(i, r0, r1, r2, r3, r4) \
       {           \
    r0 ^= r2;   \
    r4 = r2;   \
    r2 &= r0;   \
    r4 ^= r3;   \
    r2 = ~r2;      \
    r3 ^= r1;   \
    r2 ^= r3;   \
    r4 |= r0;   \
    r0 ^= r2;   \
    r3 ^= r4;   \
    r4 ^= r1;   \
    r1 &= r3;   \
    r1 ^= r0;   \
    r0 ^= r3;   \
    r0 |= r2;   \
    r3 ^= r1;   \
    r4 ^= r0;   \
            }

#define S7(i, r0, r1, r2, r3, r4) \
       {           \
    r4 = r2;   \
    r2 &= r1;   \
    r2 ^= r3;   \
    r3 &= r1;   \
    r4 ^= r2;   \
    r2 ^= r1;   \
    r1 ^= r0;   \
    r0 |= r4;   \
    r0 ^= r2;   \
    r3 ^= r1;   \
    r2 ^= r3;   \
    r3 &= r0;   \
    r3 ^= r4;   \
    r4 ^= r2;   \
    r2 &= r0;   \
    r4 = ~r4;      \
    r2 ^= r4;   \
    r4 &= r0;   \
    r1 ^= r3;   \
    r4 ^= r1;   \
            }

#define I7(i, r0, r1, r2, r3, r4) \
       {           \
    r4 = r2;   \
    r2 ^= r0;   \
    r0 &= r3;   \
    r2 = ~r2;      \
    r4 |= r3;   \
    r3 ^= r1;   \
    r1 |= r0;   \
    r0 ^= r2;   \
    r2 &= r4;   \
    r1 ^= r2;   \
    r2 ^= r0;   \
    r0 |= r2;   \
    r3 &= r4;   \
    r0 ^= r3;   \
    r4 ^= r1;   \
    r3 ^= r4;   \
    r4 |= r0;   \
    r3 ^= r2;   \
    r4 ^= r2;   \
            }

// key xor
#define KX(r, a, b, c, d, e)    {\
        a ^= subkey[4 * r + 0]; \
        b ^= subkey[4 * r + 1]; \
        c ^= subkey[4 * r + 2]; \
        d ^= subkey[4 * r + 3];}


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
