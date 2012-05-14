// See serpent.h

#include "serpent.h"

#include <cuda.h>
#include <cuda_runtime_api.h>

#ifndef SUBKEY_LENGTH
#define SUBKEY_LENGTH 132
#endif


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

/**	Decrypt a single block on the device.
 */
__device__ void serpent_cuda_decrypt_block(block128_t* block, uint32_t* subkey);


/**	Decrypt the specified array of blocks with the specified subkey through a CUDA thread.
 */
__global__ void serpent_cuda_decrypt_blocks(block128_t* cuda_blocks);


/**	Encrypt a single block on the device.
 */
__device__ void serpent_cuda_encrypt_block(block128_t* block, uint32_t* subkey);


/**	Encrypt the specified array of blocks with the specified subkey through a CUDA thread.
 */
__global__ void serpent_cuda_encrypt_blocks(block128_t* cuda_blocks);


/**	Flip the bytes of the specified 32-bit unsigned integer.
 *	@return	A 32-bit unsigned integer with the bytes mirrored.
 */
__device__ uint32_t mirror_bytes32_cu(uint32_t x);


// Constant variables must be declared with a static scope...
// Array to hold the expanded serpent key.
__device__ __constant__ uint32_t cuda_subkey[SUBKEY_LENGTH];
// The total number of blocks being decrypted by a single CUDA thread.
__device__ __constant__ int blocks_per_thread;
// The total number of blocks being decrypted in the entire CUDA kernel.
__device__ __constant__ int blocks_per_kernel;


__device__ void serpent_cuda_decrypt_block(block128_t* block, uint32_t* subkey) {
	uint32_t a, b, c, d, e;
	int j;

	// Change to little endian.
        a = mirror_bytes32_cu(block->x0);
        b = mirror_bytes32_cu(block->x1);
        c = mirror_bytes32_cu(block->x2);
        d = mirror_bytes32_cu(block->x3);

	// Decrypt the current block.
	j = 4;
	subkey += 96;
	beforeI7(KX);
	goto start;
	do
	{
		c = b;
		b = d;
		d = e;
		subkey -= 32;
		beforeI7(inverse_linear_transformation);
	start:
		beforeI7(I7); afterI7(KX);
		afterI7(inverse_linear_transformation); afterI7(I6); afterI6(KX);
		afterI6(inverse_linear_transformation); afterI6(I5); afterI5(KX);
		afterI5(inverse_linear_transformation); afterI5(I4); afterI4(KX);
		afterI4(inverse_linear_transformation); afterI4(I3); afterI3(KX);
		afterI3(inverse_linear_transformation); afterI3(I2); afterI2(KX);
		afterI2(inverse_linear_transformation); afterI2(I1); afterI1(KX);
		afterI1(inverse_linear_transformation); afterI1(I0); afterI0(KX);
	}
	while (--j != 0);

	// Restore to big endian based on algorithm-defined order.
	block->x0 = mirror_bytes32_cu(a);
	block->x1 = mirror_bytes32_cu(d);
	block->x2 = mirror_bytes32_cu(b);
	block->x3 = mirror_bytes32_cu(e);
}


__global__ void serpent_cuda_decrypt_blocks(block128_t* cuda_blocks) {
	int index = (blockIdx.x * (blockDim.x * blocks_per_thread)) + threadIdx.x;
	int i;

	// Decrypt the minimal number of blocks.
	for ( i = 0; i < blocks_per_thread; i++ ) {
		serpent_cuda_decrypt_block(&(cuda_blocks[index]), cuda_subkey);

		index += blockDim.x;
	}

	// Decrypt the extra blocks that fall outside the minimal number of blocks.
	index = ( gridDim.x * blockDim.x * blocks_per_thread ) + ((blockIdx.x * blockDim.x) + threadIdx.x); // (end of array) + (absolute thread #).
	if ( index < blocks_per_kernel ) {
		serpent_cuda_decrypt_block(&(cuda_blocks[index]), cuda_subkey);
	}
}

/*
__global__ void serpent_cuda_decrypt_blocks(block128_t* cuda_blocks) {
	int index = (blockIdx.x * blockDim.x * blocks_per_thread) + (threadIdx.x * blocks_per_thread); // (beginning of multiprocessor segment) + (segment index).
	int i;

	// Encrypted the minimal number of blocks.
	for ( i = 0; i < blocks_per_thread; i++ ) {
		serpent_cuda_decrypt_block(&(cuda_blocks[index + i]), cuda_subkey);
	}

	// Encrypt the extra blocks that fall outside the minimal number of block.s
	index = ( gridDim.x * blockDim.x * blocks_per_thread ) + ((blockIdx.x * blockDim.x) + threadIdx.x); // (end of array) + (absolute thread #).
	if ( index < blocks_per_kernel ) {
		serpent_cuda_decrypt_block(&(cuda_blocks[index]), cuda_subkey);
	}
}
*/

__device__ void serpent_cuda_encrypt_block(block128_t* block, uint32_t* subkey) {
	uint32_t a, b, c, d, e;
	int j;

	// Change to little endian.
	a = mirror_bytes32_cu(block->x0);
	b = mirror_bytes32_cu(block->x1);
	c = mirror_bytes32_cu(block->x2);
	d = mirror_bytes32_cu(block->x3);

	// Encrypt the current block.
	j = 1;
	do {
		beforeS0(KX); beforeS0(S0); afterS0(linear_transformation);
		afterS0(KX); afterS0(S1); afterS1(linear_transformation);
		afterS1(KX); afterS1(S2); afterS2(linear_transformation);
		afterS2(KX); afterS2(S3); afterS3(linear_transformation);
		afterS3(KX); afterS3(S4); afterS4(linear_transformation);
		afterS4(KX); afterS4(S5); afterS5(linear_transformation);
		afterS5(KX); afterS5(S6); afterS6(linear_transformation);
		afterS6(KX); afterS6(S7);

		if (j == 4)
			break;

		++j;
		c = b;
		b = e;
		e = d;
		d = a;
		a = e;
		subkey += 32;
		beforeS0(linear_transformation);
	} while (1);
	afterS7(KX);

	// Restore to big endian based on algorithm-defined order.
	block->x0 = mirror_bytes32_cu(d);
	block->x1 = mirror_bytes32_cu(e);
	block->x2 = mirror_bytes32_cu(b);
	block->x3 = mirror_bytes32_cu(a);
}


__global__ void serpent_cuda_encrypt_blocks( block128_t* cuda_blocks ) {
        int index = (blockIdx.x * (blockDim.x * blocks_per_thread)) + threadIdx.x;
        int i;

        // Encrypt the minimal number of blocks.
        for ( i = 0; i < blocks_per_thread; i++ ) {
                // Encrypt the block.
                serpent_cuda_encrypt_block(&(cuda_blocks[index]), cuda_subkey);

                // Increment the index.
                index += blockDim.x;
        }

        // Encrypt the extra blocks that fall outside the minimal number of block.s
        index = (gridDim.x * (blockDim.x * blocks_per_thread)) + ((blockIdx.x * blockDim.x) + threadIdx.x); // (end of array) + (absolute thread #).
        if ( index < blocks_per_kernel ) {
                serpent_cuda_encrypt_block(&(cuda_blocks[index]), cuda_subkey);
        }
}

/* A better attempt at stronger global memory coalescing. Still did not turn out well.
#define UINT32_PER_BLOCK128 4

__device__ void serpent_cuda_encrypt_block(uint32_t* shared_blocks, int shared_index, uint32_t* subkey);
__global__ void serpent_cuda_encrypt_blocks(uint32_t* cuda_blocks);

__global__ void serpent_cuda_encrypt_blocks( uint32_t* cuda_blocks ) {
	int threads_per_multiprocessor = blockDim.x;
        int cache_index = (blockIdx.x * (UINT32_PER_BLOCK128 * threads_per_multiprocessor * blocks_per_thread));
        int i;

        // Encrypt the minimal number of blocks.
        for ( i = 0; i < blocks_per_thread; i++ ) {
                // Encrypt the blocks at the cache index
                serpent_cuda_encrypt_block(&(cuda_blocks[cache_index]), cuda_subkey, threads_per_multiprocessor);

		// Adjust cache index value.
		cache_index += (threads_per_multiprocessor * UINT32_PER_BLOCK128);
	}

	// Encrypt the extra blocks that fall outside the minimal number of block.
	// NOTE: DOES NOT WORK and is incomplete.
	//cache_index = (gridDim.x * (threads_per_multiprocessor * blocks_per_thread)) + (blockIdx.x * threads_per_multiprocessor); // (end of array + multiprocessor block).
	//if ( cache_index > blocks_per_kernel) {
	//	return;
	//}
	//else if ( (cache_index + threads_per_multiprocessor) > blocks_per_kernel ) {
	//	if ( threadIdx.x + cache_index > blocks_per_kernel ) {
	//		return;
	//	}
	//	threads_per_multiprocessor = (blocks_per_kernel - cache_index);
	//}
}
__device__ void serpent_cuda_encrypt_block(uint32_t* global_blocks, uint32_t* subkey, int threads_per_multiprocessor) {
	// Array that allows collaborative loading of blocks into shared memory.
	extern __shared__ uint32_t shared_blocks[];
	uint32_t a, b, c, d, e;
	int index = threadIdx.x;
	int j;

	// Collaboratively load blocks into shared memory.
	shared_blocks[index] = mirror_bytes32_cu(global_blocks[index]);
	index += threads_per_multiprocessor;
	shared_blocks[index] = mirror_bytes32_cu(global_blocks[index]);
	index += threads_per_multiprocessor;
	shared_blocks[index] = mirror_bytes32_cu(global_blocks[index]);
	index += threads_per_multiprocessor;
	shared_blocks[index] = mirror_bytes32_cu(global_blocks[index]);
	index -= (threads_per_multiprocessor * 3);
	__syncthreads();

	// Read from shared memory.
	index *= UINT32_PER_BLOCK128;
	a = shared_blocks[index];
	b = shared_blocks[index+1];
	c = shared_blocks[index+2];
	d = shared_blocks[index+3];

	// Encrypt the current block.
	j = 1;
	do {
		beforeS0(KX); beforeS0(S0); afterS0(linear_transformation);
		afterS0(KX); afterS0(S1); afterS1(linear_transformation);
		afterS1(KX); afterS1(S2); afterS2(linear_transformation);
		afterS2(KX); afterS2(S3); afterS3(linear_transformation);
		afterS3(KX); afterS3(S4); afterS4(linear_transformation);
		afterS4(KX); afterS4(S5); afterS5(linear_transformation);
		afterS5(KX); afterS5(S6); afterS6(linear_transformation);
		afterS6(KX); afterS6(S7);

		if (j == 4)
			break;

		++j;
		c = b;
		b = e;
		e = d;
		d = a;
		a = e;
		subkey += 32;
		beforeS0(linear_transformation);
	} while (1);
	afterS7(KX);

	// Write blocks back to global memory.
	global_blocks[index] = mirror_bytes32_cu(d);
	global_blocks[index+1] = mirror_bytes32_cu(e);
	global_blocks[index+2] = mirror_bytes32_cu(b);
	global_blocks[index+3] = mirror_bytes32_cu(a);
	__syncthreads();
}

serpent_cuda_encrypt_blocks<<<multiprocessor_count, thread_count, (sizeof(block128_t) * thread_count)>>>((uint32_t*)cuda_blocks);

 */

__device__ uint32_t mirror_bytes32_cu(uint32_t x) {
	uint32_t out;

	// Change to Little Endian.
	out = (uint8_t) x;
       	out <<= 8; out |= (uint8_t) (x >> 8);
	out <<= 8; out |= (uint8_t) (x >> 16);
	out = (out << 8) | (uint8_t) (x >> 24);

	// Return out.
	return out;
}


extern "C"
inline int serpent_cuda_allocate_buffer(size_t free_global_memory, size_t total_global_memory, int block_count, int multiprocessor_count, int thread_count, block128_t** cuda_blocks, size_t* used_global_memory_output, int* blocks_per_kernel_output, int* blocks_per_thread_output, int* buffer_allocation_attempts_output) {
	cudaError_t cuda_error;
	int blocks_per_kernel;
	int blocks_per_thread;
	int buffer_allocation_attempts;

	// Validate parameters.
	#ifdef DEBUG_SERPENT
	if ( free_global_memory < 1 ) {
		fprintf(stderr, "Free global memory less than 1.\n");
		return -1;
	}
	else if ( cuda_blocks == NULL ) {
		fprintf(stderr, "cuda_blocks was NULL.\n");
		return -1;
	}
	else if ( used_global_memory_output == NULL ) {
		fprintf(stderr, "used_global_memory_output was NULL.\n");
		return -1;
	}
	else if ( blocks_per_kernel_output == NULL ) {
		fprintf(stderr, "blocks_per_kernel_output was NULL.\n");
		return -1;
	}
	else if ( blocks_per_thread_output == NULL ) {
		fprintf(stderr, "blocks_per_thread_output was NULL.\n");
		return -1;
	}
	#endif

	// Try to allocate blocks.
	buffer_allocation_attempts = 1;
	while ( true ) {
		// Subtract a small portion of global memory.
		free_global_memory -= total_global_memory * SERPENT_CUDA_MEMORY_MULTIPLIER;
		if ( free_global_memory <= 0 ) {
			fprintf(stderr, "No memory for blocks available.\n");
			return -1;
		}

		// Calculate number of blocks per thread.
		blocks_per_kernel = free_global_memory / sizeof(block128_t);
		if ( blocks_per_kernel > block_count ) {
			blocks_per_kernel = block_count;
		}
		blocks_per_thread = (blocks_per_kernel / multiprocessor_count) / thread_count;

		// Attempt to allocate memory on the GPU.
		cuda_error = cudaMalloc( (void**)cuda_blocks, (int)(sizeof(block128_t) * blocks_per_kernel) );
		if ( cuda_error == cudaSuccess ) { // Success! Exit this loop.
			break;
		}
		else if ( cuda_error != cudaErrorMemoryAllocation ) {
			fprintf(stderr, "Unable to malloc blocks: %s.\n", cudaGetErrorString(cuda_error));
			return -1;
		}

		buffer_allocation_attempts++;
	}

	// Assign output parameters.
	(*used_global_memory_output) = free_global_memory;
	(*blocks_per_kernel_output) = blocks_per_kernel;
	(*blocks_per_thread_output) = blocks_per_thread;
	if ( buffer_allocation_attempts_output != NULL ) {
		(*buffer_allocation_attempts_output) = buffer_allocation_attempts;
	}

	// Return success.
	return 0;
}


extern "C"
int serpent_cuda_decrypt_cu(uint32_t* subkey, block128_t* blocks, int block_count, size_t* buffer_size) {
	// Total number of registers taken up by a single CUDA thread.
	const int REGISTERS_PER_THREAD = 8;
	block128_t* cuda_blocks;
	cudaError_t cuda_error;
	size_t total_global_memory;
	size_t free_global_memory;
	int blocks_per_kernel;
	int blocks_per_thread;
	int buffer_allocation_attempts;
	int count; 
	int device_number;
	int kernel_invocation_attempts;
	int multiprocessor_count;
	int temp;
	int thread_count;
	int i;

	// Validate parameters.
	#ifdef DEBUG_SERPENT
	if ( subkey == NULL ) {
		fprintf(stderr, "subkey was NULL.\n");
		return -1;
	}
	else if ( blocks == NULL ) {
		fprintf(stderr, "blocks was NULL.\n");
		return -1;
	}
	else if ( block_count < 1 ) {
		fprintf(stderr, "block_count was less than 1.\n");
		return -1;
	}
	else if ( buffer_size == NULL ) {
		fprintf(stderr, "buffer_size was NULL.\n");
		return -1;
	}
	#endif

	// Get the number of devices.
	cuda_error = cudaGetDeviceCount( &count );
	if ( cuda_error != cudaSuccess ) {
		fprintf(stderr, "Unable to get device count: %s.\n", cudaGetErrorString(cuda_error));
		return -1;
	}
	else if ( count == 0 ) {
		fprintf(stderr, "No CUDA-capable devices found.\n");
		return -1;
	}

	// Calculate multiprocessor and thread count.
	device_number = 0;
	if ( cuda_get_block_and_thread_count_max(device_number, REGISTERS_PER_THREAD, &multiprocessor_count, &thread_count) == -1 ) {
		fprintf(stderr, "Unable to get multiprocessor and thread count.\n");
		return -1;
	}

	// Move subkey to constant memory.
	cuda_error = cudaMemcpyToSymbol( "cuda_subkey", subkey, sizeof(uint32_t) * SUBKEY_LENGTH);
	if ( cuda_error != cudaSuccess ) {
		fprintf(stderr, "Unable to copy subkey to constant memory: %s.\n", cudaGetErrorString(cuda_error));
		return -1;
	}

	// Calculate the amount of global memory available for blocks.
	cuda_error = cudaMemGetInfo(&free_global_memory, &total_global_memory);
	if ( cuda_error != cudaSuccess ) {
		fprintf(stderr, "Unable to get memory information: %s.\n", cudaGetErrorString(cuda_error));
		return -1;
	}

	// Allocate blocks buffer on the GPU.
	if ( serpent_cuda_allocate_buffer(free_global_memory, total_global_memory, block_count, multiprocessor_count, thread_count, &cuda_blocks, &free_global_memory, &blocks_per_kernel, &blocks_per_thread, &buffer_allocation_attempts) == -1 ) {
		fprintf(stderr, "Unable to allocate initial buffer.\n");
		return -1;
	}

	// Decrypt the blocks.
	i = 0;
	while (i < block_count) {
		// Corner case.
		if ( i + blocks_per_kernel > block_count ) {
			blocks_per_kernel = block_count - i;
			blocks_per_thread = blocks_per_kernel / multiprocessor_count / thread_count;
		}

		// Run the algorithm.
		kernel_invocation_attempts = 1;
		while( true ) {
			// Move blocks to global memory.
			cuda_error = cudaMemcpy( cuda_blocks, &(blocks[i]), sizeof(block128_t) * blocks_per_kernel, cudaMemcpyHostToDevice );
			if ( cuda_error != cudaSuccess ) {
				fprintf(stderr, "Unable to memcopy blocks: %s.\n", cudaGetErrorString(cuda_error));
				return -1;
			}

			// Copy blocks per thread to constant memory.
			cuda_error = cudaMemcpyToSymbol( "blocks_per_thread", &blocks_per_thread, sizeof(int));
			if ( cuda_error != cudaSuccess ) {
				fprintf(stderr, "Unable to copy blocks_per_thread to constant memory: %s.\n", cudaGetErrorString(cuda_error));
				return -1;
			}

			// Copy blocks per kernel to constant memory.
			cuda_error = cudaMemcpyToSymbol( "blocks_per_kernel", &blocks_per_kernel, sizeof(int));
			if ( cuda_error != cudaSuccess ) {
				fprintf(stderr, "Unable to copy blocks_per_kernel to constant memory: %s.\n", cudaGetErrorString(cuda_error));
				return -1;
			}

			// Run encryption.
			serpent_cuda_decrypt_blocks<<<multiprocessor_count, thread_count>>>(cuda_blocks);
			cuda_error = cudaGetLastError();
			if ( cuda_error == cudaSuccess ) { // Successful run.
				break;
			}
			else if ( cuda_error != cudaErrorMemoryAllocation ) { // Unexpected error.
				fprintf(stderr, "ERROR invoking the kernel: %s, %i.\n", cudaGetErrorString(cuda_error), cuda_error);
				return -1;
			}

			// Free the old blocks buffer.
			cudaFree(cuda_blocks);

			// Allocate a new blocks buffer.
			if ( serpent_cuda_allocate_buffer(free_global_memory, total_global_memory, block_count, multiprocessor_count, thread_count, &cuda_blocks, &free_global_memory, &blocks_per_kernel, &blocks_per_thread, &temp) == -1 ) {
				fprintf(stderr, "Unable to reallocate blocks buffer.\n");
				return -1;
			}
			buffer_allocation_attempts += temp;

			kernel_invocation_attempts++;
		}

		// Get blocks from global memory.
		cuda_error = cudaMemcpy( &(blocks[i]), cuda_blocks, sizeof(block128_t) * blocks_per_kernel, cudaMemcpyDeviceToHost );
		if ( cuda_error != cudaSuccess ) {
			fprintf(stderr, "Unable to retrieve blocks: %s.\n", cudaGetErrorString(cuda_error));
			return -1;
		}
	
		// Increment i by the number of blocks processed.
		i += blocks_per_kernel;
	}

	// Free blocks from global memory.
	cudaFree(cuda_blocks);

	// Assign output parameters.
	(*buffer_size) = free_global_memory;

	// TODO: Add these as output parameters.
	//fprintf(stderr, "Buffer allocation attempts: %i.\nKernel invocation attempts: %i.\n",
	//	buffer_allocation_attempts, kernel_invocation_attempts);

	// Return success.
	return 0;
}


extern "C"
int serpent_cuda_encrypt_cu(uint32_t* subkey, block128_t* blocks, int block_count, size_t* buffer_size) {
	// Maximum total number of registers taken up by a single CUDA thread.
	// This variable will need to be manually calculated and updated if
	// the algorithm implementation changes (but if you know of a way
	// to proceedurally do this, please, feel free...).
	const int REGISTERS_PER_THREAD = 8;
	//cudaDeviceProp cuda_device;
	block128_t* cuda_blocks;
	cudaError_t cuda_error;
	size_t total_global_memory;
	size_t free_global_memory;
	int buffer_allocation_attempts;
	int kernel_invocation_attempts;
	int blocks_per_kernel;
	int blocks_per_thread;
	int count;
	int device_number;
	int multiprocessor_count;
	int temp;
	int thread_count;
	int i;

	// Validate parameters.
	#ifdef DEBUG_SERPENT
	if ( subkey == NULL ) {
		fprintf(stderr, "subkey was NULL.\n");
		return -1;
	}
	else if ( blocks == NULL ) {
		fprintf(stderr, "blocks was NULL.\n");
		return -1;
	}
	else if ( block_count < 1 ) {
		fprintf(stderr, "block_count was less than 1.\n");
		return -1;
	}
	else if ( buffer_size == NULL ) {
		fprintf(stderr, "buffer_size was NULL.\n");
		return -1;
	}
	#endif

	// Get the number of devices.
	cuda_error = cudaGetDeviceCount( &count );
	if ( cuda_error != cudaSuccess ) {
		fprintf(stderr, "Unable to get device count: %s.\n", cudaGetErrorString(cuda_error));
		return -1;
	}
	else if ( count == 0 ) {
		fprintf(stderr, "No CUDA-capable devices found.\n");
		return -1;
	}

	// Calculate the number of multiprocessors and threads to launch.
	device_number = 0;
	if ( cuda_get_block_and_thread_count_max(device_number, REGISTERS_PER_THREAD, &multiprocessor_count, &thread_count) == -1 ) {
		fprintf(stderr, "Unable to get max thread count.\n");
		return -1;
	}

	// Copy the subkey to constant memory.
	cuda_error = cudaMemcpyToSymbol( "cuda_subkey", subkey, sizeof(uint32_t) * SUBKEY_LENGTH);
	if ( cuda_error != cudaSuccess ) {
		fprintf(stderr, "Unable to copy subkey to constant memory: %s.\n", cudaGetErrorString(cuda_error));
		return -1;
	}

	// Calculate the amount of global memory available for blocks.
	cuda_error = cudaMemGetInfo(&free_global_memory, &total_global_memory);
	if ( cuda_error != cudaSuccess ) {
		fprintf(stderr, "Unable to get memory information: %s.\n", cudaGetErrorString(cuda_error));
		return -1;
	}

	// Allocate blocks buffer on the GPU.
	if ( serpent_cuda_allocate_buffer(free_global_memory, total_global_memory, block_count, multiprocessor_count, thread_count, &cuda_blocks, &free_global_memory, &blocks_per_kernel, &blocks_per_thread, &buffer_allocation_attempts) == -1 ) {
		fprintf(stderr, "Unable to allocate initial buffer.\n");
		return -1;
	}
	
	// Encrypt the blocks.
	i = 0;
	while (i < block_count) {
		// Corner case.
		if ( i + blocks_per_kernel > block_count ) {
			blocks_per_kernel = block_count - i;
			blocks_per_thread = blocks_per_kernel / multiprocessor_count / thread_count;
		}

		// Run the algorithm.
		kernel_invocation_attempts = 1;
		while ( true ) {
			// Move blocks to global memory.
			cuda_error = cudaMemcpy( cuda_blocks, &(blocks[i]), sizeof(block128_t) * blocks_per_kernel, cudaMemcpyHostToDevice );
			if ( cuda_error != cudaSuccess ) {
				fprintf(stderr, "Unable to memcopy blocks: %s.\n", cudaGetErrorString(cuda_error));
				return -1;
			}

			// Copy blocks per thread to constant memory.
			cuda_error = cudaMemcpyToSymbol( "blocks_per_thread", &blocks_per_thread, sizeof(int));
			if ( cuda_error != cudaSuccess ) {
				fprintf(stderr, "Unable to copy blocks_per_thread to constant memory: %s.\n", cudaGetErrorString(cuda_error));
				return -1;
			}

			// Copy blocks per kernel to constant memory.
			cuda_error = cudaMemcpyToSymbol( "blocks_per_kernel", &blocks_per_kernel, sizeof(int));
			if ( cuda_error != cudaSuccess ) {
				fprintf(stderr, "Unable to copy blocks_per_kernel to constant memory: %s.\n", cudaGetErrorString(cuda_error));
				return -1;
			}

			// Run encryption.
			serpent_cuda_encrypt_blocks<<<multiprocessor_count, thread_count>>>(cuda_blocks);
			cuda_error = cudaGetLastError();
			if ( cuda_error == cudaSuccess ) { // Successful run.
				break;
			}
			else if ( cuda_error != cudaErrorMemoryAllocation ) { // Unexpected error.
				fprintf(stderr, "ERROR invoking the kernel: %s, %i.\n", cudaGetErrorString(cuda_error), cuda_error);
				return -1;
			}

			// Free the old blocks buffer.
			cudaFree(cuda_blocks);

			// Allocate a new blocks buffer.
			if ( serpent_cuda_allocate_buffer(free_global_memory, total_global_memory, block_count, multiprocessor_count, thread_count, &cuda_blocks, &free_global_memory, &blocks_per_kernel, &blocks_per_thread, &temp) == -1 ) {
				fprintf(stderr, "Unable to reallocate blocks buffer.\n");
				return -1;
			}
			buffer_allocation_attempts += temp;

			kernel_invocation_attempts++;
		}
		
		// Get blocks from global memory.
		cuda_error = cudaMemcpy( &(blocks[i]), cuda_blocks, sizeof(block128_t) * blocks_per_kernel, cudaMemcpyDeviceToHost );
		if ( cuda_error != cudaSuccess ) {
			fprintf(stderr, "Unable to retrieve blocks: %s.\n", cudaGetErrorString(cuda_error));
			return -1;
		}
	
		// Increment i by the number of blocks processed.
		i += blocks_per_kernel;
	}

	// Free blocks from global memory.
	cudaFree(cuda_blocks);

	// Assign output parameters.
	(*buffer_size) = free_global_memory;

	// TODO: Makes this as function output.
	//fprintf(stderr, "Buffer allocation attempts: %i.\nKernel invocation attempts: %i.\n",
		//buffer_allocation_attempts, kernel_invocation_attempts);

	// Return success.
	return 0;
}
