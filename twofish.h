/*
 ---------------------------------------------------------------------------
 Copyright (c) 1999, Dr Brian Gladman, Worcester, UK.   All rights reserved.

 LICENSE TERMS

 The free distribution and use of this software is allowed (with or without
 changes) provided that:

  1. source code distributions include the above copyright notice, this
     list of conditions and the following disclaimer;

  2. binary distributions include the above copyright notice, this list
     of conditions and the following disclaimer in their documentation;

  3. the name of the copyright holder is not used to endorse products
     built using this software without specific written permission.

 DISCLAIMER

 This software is provided 'as is' with no explicit or implied warranties
 in respect of its properties, including, but not limited to, correctness
 and/or fitness for purpose.
 ---------------------------------------------------------------------------

 My thanks to Doug Whiting and Niels Ferguson for comments that led
 to improvements in this implementation.

 Issue Date: 14th January 1999
*/

/* Subsequently adapted for TrueCrypt (copyrighted by its respectful owners)... 
   and then subsequently re-adapted for CUDA benchmarking. */

// A copy of the TrueCrypt license is contained in the LICENSE file.


#ifndef Twofish_H
#define Twofish_H

/* DEBUG ... because I am debugging! */
#define DEBUG_TWOFISH

#ifdef DEBUG
#define DEBUG_TWOFISH
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


// Percentage of global memory to subtract per iteration when trying to allocate
// a memory buffer for blocks in CUDA.
#define TWOFISH_CUDA_MEMORY_MULTIPLIER 0.001


// Rotate the bits in the specified number x left by the specified number n.
#define rotl_fixed(x, n)   (((x) << (n)) | ((x) >> (32 - (n))))
// Rotate the bits in the specified number x right by the specified number n.
#define rotr_fixed(x, n)   (((x) >> (n)) | ((x) << (32 - (n))))


// Structure that holds key-specific data that the Twofish algorithm uses.
// This appears to be known in the Advanced Encryption Standard (AES)
// submission as "The Key Schedule".
typedef struct {
	// Appears to be the 40 words of the expanded key.
	uint32_t l_key[40];
	// Appears to be the 4 key-dependent S-boxes used in the 'g function'.
	uint32_t s_key[4];
	// Appears to be another component derived from the key used in the algorithm.
	uint32_t mk_tab[4 * 256];
	// Appears to be the key length as a multiple of 64 bits (rounded down).
	uint32_t k_len;
} twofish_instance_t;


// BEGIN wall of Twofish definitions.
#define G_M 0x0169

extern uint8_t tab_5b[4];
extern uint8_t tab_ef[4];

#define ffm_01(x)   (x)
#define ffm_5b(x)   ((x) ^ ((x) >> 2) ^ tab_5b[(x) & 3])
#define ffm_ef(x)   ((x) ^ ((x) >> 1) ^ ((x) >> 2) ^ tab_ef[(x) & 3])

extern uint8_t ror4[16];
extern uint8_t ashx[16];

extern uint8_t qt0[2][16];
extern uint8_t qt1[2][16];
extern uint8_t qt2[2][16];
extern uint8_t qt3[2][16];

uint8_t twofish_qp(const uint32_t n, const uint8_t x);

// Stuff to do with Q tables.
extern uint8_t q_tab[2][256];
#define q(n,x) q_tab[n][x]
void twofish_gen_qtab(void);

// Stuff to do with the M table.
extern uint32_t m_tab[4][256];
void twofish_gen_mtab(void);
#define mds(n,x) m_tab[n][x]

/* The (12,8) Reed Soloman code has the generator polynomial

  g(x) = x^4 + (a + 1/a) * x^3 + a * x^2 + (a + 1/a) * x + 1

where the coefficients are in the finite field GF(2^8) with a
modular polynomial a^8 + a^6 + a^3 + a^2 + 1. To generate the
remainder we have to start with a 12th order polynomial with our
eight input bytes as the coefficients of the 4th to 11th terms. 
That is:

  m[7] * x^11 + m[6] * x^10 ... + m[0] * x^4 + 0 * x^3 +... + 0
  
We then multiply the generator polynomial by m[7] * x^7 and subtract
it - xor in GF(2^8) - from the above to eliminate the x^7 term (the 
artihmetic on the coefficients is done in GF(2^8). We then multiply 
the generator polynomial by x^6 * coeff(x^10) and use this to remove
the x^10 term. We carry on in this way until the x^4 term is removed
so that we are left with:

  r[3] * x^3 + r[2] * x^2 + r[1] 8 x^1 + r[0]

which give the resulting 4 bytes of the remainder. This is equivalent 
to the matrix multiplication in the Twofish description but much faster 
to implement.

*/

#define G_MOD   0x0000014d

uint32_t twofish_mds_rem(uint32_t p0, uint32_t p1);

#define extract_byte(x,n)   ((uint8_t)((x) >> (8 * n)))

uint32_t twofish_h_fun(twofish_instance_t* instance, const uint32_t x, const uint32_t key[]);

#define q20(x)  q(0,q(0,x) ^ extract_byte(key[1],0)) ^ extract_byte(key[0],0)
#define q21(x)  q(0,q(1,x) ^ extract_byte(key[1],1)) ^ extract_byte(key[0],1)
#define q22(x)  q(1,q(0,x) ^ extract_byte(key[1],2)) ^ extract_byte(key[0],2)
#define q23(x)  q(1,q(1,x) ^ extract_byte(key[1],3)) ^ extract_byte(key[0],3)

#define q30(x)  q(0,q(0,q(1, x) ^ extract_byte(key[2],0)) ^ extract_byte(key[1],0)) ^ extract_byte(key[0],0)
#define q31(x)  q(0,q(1,q(1, x) ^ extract_byte(key[2],1)) ^ extract_byte(key[1],1)) ^ extract_byte(key[0],1)
#define q32(x)  q(1,q(0,q(0, x) ^ extract_byte(key[2],2)) ^ extract_byte(key[1],2)) ^ extract_byte(key[0],2)
#define q33(x)  q(1,q(1,q(0, x) ^ extract_byte(key[2],3)) ^ extract_byte(key[1],3)) ^ extract_byte(key[0],3)

#define q40(x)  q(0,q(0,q(1, q(1, x) ^ extract_byte(key[3],0)) ^ extract_byte(key[2],0)) ^ extract_byte(key[1],0)) ^ extract_byte(key[0],0)
#define q41(x)  q(0,q(1,q(1, q(0, x) ^ extract_byte(key[3],1)) ^ extract_byte(key[2],1)) ^ extract_byte(key[1],1)) ^ extract_byte(key[0],1)
#define q42(x)  q(1,q(0,q(0, q(0, x) ^ extract_byte(key[3],2)) ^ extract_byte(key[2],2)) ^ extract_byte(key[1],2)) ^ extract_byte(key[0],2)
#define q43(x)  q(1,q(1,q(0, q(1, x) ^ extract_byte(key[3],3)) ^ extract_byte(key[2],3)) ^ extract_byte(key[1],3)) ^ extract_byte(key[0],3)

void twofish_gen_mk_tab(twofish_instance_t* instance, uint32_t key[]);

// For encrypt block.
#define f_rnd(i)								\
	t1 = g1_fun(blk[1]); t0 = g0_fun(blk[0]);			   	\
	blk[2] = rotr_fixed(blk[2] ^ (t0 + t1 + l_key[4 * (i) + 8]), 1);	\
	blk[3] = rotl_fixed(blk[3], 1) ^ (t0 + 2 * t1 + l_key[4 * (i) + 9]);	\
	t1 = g1_fun(blk[3]); t0 = g0_fun(blk[2]);				\
	blk[0] = rotr_fixed(blk[0] ^ (t0 + t1 + l_key[4 * (i) + 10]), 1);	\
	blk[1] = rotl_fixed(blk[1], 1) ^ (t0 + 2 * t1 + l_key[4 * (i) + 11])

// For decrypt block.
#define i_rnd(i)								\
	t1 = g1_fun(blk[1]); t0 = g0_fun(blk[0]);			   	\
	blk[2] = rotl_fixed(blk[2], 1) ^ (t0 + t1 + l_key[4 * (i) + 10]);	\
	blk[3] = rotr_fixed(blk[3] ^ (t0 + 2 * t1 + l_key[4 * (i) + 11]), 1);	\
	t1 = g1_fun(blk[3]); t0 = g0_fun(blk[2]);			   	\
	blk[0] = rotl_fixed(blk[0], 1) ^ (t0 + t1 + l_key[4 * (i) +  8]);	\
	blk[1] = rotr_fixed(blk[1] ^ (t0 + 2 * t1 + l_key[4 * (i) +  9]), 1)

// For encrypt and decrypt (sure seems to be a lot of fun in this code ;))
#define g0_fun(x) ( mk_tab[0 + 4*extract_byte(x,0)] ^ mk_tab[1 + 4*extract_byte(x,1)] \
			  ^ mk_tab[2 + 4*extract_byte(x,2)] ^ mk_tab[3 + 4*extract_byte(x,3)] )
#define g1_fun(x) ( mk_tab[0 + 4*extract_byte(x,3)] ^ mk_tab[1 + 4*extract_byte(x,0)] \
			  ^ mk_tab[2 + 4*extract_byte(x,1)] ^ mk_tab[3 + 4*extract_byte(x,2)] )
// END wall of Twofish definitions.


/**	Run the specified array of 128-bit blocks through the twofish encryption algorithm.
 *	@out	buffer_size: Size of the global memory buffer used (only for CUDA).
 *	@return NULL on success, exception_t* on failure.
 */
exception_t* twofish(key256_t* user_key, block128_t* blocks, int block_count, enum mode mode, enum encryption encryption, size_t* buffer_size);


/**	Decrypt the specified array of 128-bit blocks through the CUDA twofish algorithm.
 *	@return	NULL on success, exception_t* on failure.
 */
exception_t* twofish_cuda_decrypt(key256_t* user_key, block128_t* blocks, int block_count, size_t* buffer_size);


/**	Encrypt the specified array of 128-bit blocks through the CUDA twofish algorithm.
 *	@return	NULL on success, exception_t* on failure.
 */
exception_t* twofish_cuda_encrypt(key256_t* user_key, block128_t* blocks, int block_count, size_t* buffer_size);


/**	Private inner function to prevent linking errors because nvcc does not like external libraries.
 *	@return 0 on success, -1 on failure.
 */
#ifdef __cplusplus
extern "C"
#endif
int twofish_cuda_decrypt_cu(twofish_instance_t* instance, block128_t* blocks, int block_count, size_t* buffer_size);


/**	Private inner function to prevent linking errors because nvcc does not like external libraries.
 *	@return 0 on success, -1 on failure.
 */
#ifdef __cplusplus
extern "C"
#endif
int twofish_cuda_encrypt_cu(twofish_instance_t* instance, block128_t* blocks, int block_count, size_t* buffer_size);


/**	Private function to decrypt a single block of twofish.
 */
void twofish_decrypt_block(block128_t* block, twofish_instance_t* instance);


/**	Private function to encrypt a single block of twofish.
 */
void twofish_encrypt_block(block128_t* block, twofish_instance_t* instance);


/**	Private function that generates the instance (key-specific data) for the twofish encryption algorithm.
 *	@return NULL on success, exception_t* on failure.
 */
exception_t* twofish_instance_init(twofish_instance_t* instance, key256_t* user_key);


/**	Debug function that pritns the twofish instance to stderr.
 *	@return NULL on success, exception_t* on failure.
 */
exception_t* twofish_instance_print(twofish_instance_t* instance);


/**	Decrypt the specified array of 128-bit blocks in parallel through the twofish encryption algorithm.
 *	@return NULL on success, exception_t* on failure.
 */
exception_t* twofish_parallel_decrypt(key256_t* user_key, block128_t* blocks, int block_count);


/**	Encrypt the specified array of 128-bit blocks in parallel through the twofish encryption algorithm.
 *	@return NULL on success, exception_t* on failure.
 */
exception_t* twofish_parallel_encrypt(key256_t* user_key, block128_t* blocks, int block_count);


/**	Decrypt the specified array of 128-bit blocks serially through the twofish encryption algorithm.
 *	@return NULL on success, exception_t* on failure.
 */
exception_t* twofish_serial_decrypt(key256_t* user_key, block128_t* blocks, int block_count);


/**	Encrypt the specified array of 128-bit blocks serially through the twofish encryption algorithm.
 *	@return	NULL on success, exception_t* on failure.
 */
exception_t* twofish_serial_encrypt(key256_t* user_key, block128_t* blocks, int block_count);


#endif // twofish_H
