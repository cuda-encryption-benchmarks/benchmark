/*
 ---------------------------------------------------------------------------
 Copyright (c) 1998-2007, Brian Gladman, Worcester, UK. All rights reserved.

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
 Issue Date: 20/12/2007
*/

/* Adapted for CUDA benchmarking March 2012. */


#ifndef aes_H
#define aes_H


#include "block128.h"
#include "ccc/ccc.h"
#include "key.h"
#include "typedef.h"


/**	Run the specified array of 128-bit blocks through the AES encryption algorithm.
 *	@return	NULL on success, exception_t* on failure.
 */
exception_t* aes(key256_t* key, block128_t* blocks, int block_count, enum mode mode, enum encryption encryption);


/**	Decrypt the specified array of 128-bit blocks through the CUDA AES algorithm.
 *	@return	NULL on success, exception_t* on failure.
 */
exception_t* aes_cuda_decrypt(key256_t* key, block128_t* blocks, int block_count);


/**	Encrypt the specified array of 128-bit blocks through the CUDA AES algorithm.
 *	@return	NULL on success, exception_t* on failure.
 */
exception_t* aes_cuda_encrypt(key256_t* key, block128_t* blocks, int block_count);


/**	Decrypt the specified array of 128-bit blocks through the parallel AES algorithm.
 *	@return	NULL on success, exception_t* on failure.
 */
exception_t* aes_parallel_decrypt(key256_t* key, block128_t* blocks, int block_count);


/**	Encrypt the specified array of 128-bit blocks through the parallel AES algorithm.
 *	@return NULL on success, exception_t* on failure.
 */
exception_t* aes_parallel_encrypt(key256_t* key, block128_t* blocks, int block_count);


/**	Decrypt the specified array of 128-bit blocks through the serial AES algorithm.
 *	@return	NULL on success, exception_t* on failure.
 */
exception_t* aes_serial_decrypt(key256_t* key, block128_t* blocks, int block_count);


/**	Encrypt the specified array of 128-bit blocks through the serial AES algorithm.
 *	@return	NULL on success, exception_t* on failure.
 */
exception_t* aes_serial_encrypt(key256_t* key, block128_t* blocks, int block_count);


#endif // aes_H
