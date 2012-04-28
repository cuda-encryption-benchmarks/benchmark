
#include "twofish.h"

typedef struct
{
	uint32_t l_key[40];
	uint32_t s_key[4];
	uint32_t mk_tab[4 * 256];
	uint32_t k_len;
} TwofishInstance;

#define extract_byte(x,n)   ((uint8_t)((x) >> (8 * n)))

#define rotr(x,n) (((x)>>(n))|((x)<<(32-(n))))
#define rotl(x,n) (((x)<<(n))|((x)>>(32-(n))))


/* finite field arithmetic for GF(2**8) with the modular    */
/* polynomial x^8 + x^6 + x^5 + x^3 + 1 (0x169)             */

#define G_M 0x0169

static uint8_t  tab_5b[4] = { 0, G_M >> 2, G_M >> 1, (G_M >> 1) ^ (G_M >> 2) };
static uint8_t  tab_ef[4] = { 0, (G_M >> 1) ^ (G_M >> 2), G_M >> 1, G_M >> 2 };

#define ffm_01(x)    (x)
#define ffm_5b(x)   ((x) ^ ((x) >> 2) ^ tab_5b[(x) & 3])
#define ffm_ef(x)   ((x) ^ ((x) >> 1) ^ ((x) >> 2) ^ tab_ef[(x) & 3])

static uint8_t ror4[16] = { 0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15 };
static uint8_t ashx[16] = { 0, 9, 2, 11, 4, 13, 6, 15, 8, 1, 10, 3, 12, 5, 14, 7 };

static uint8_t qt0[2][16] = 
{   { 8, 1, 7, 13, 6, 15, 3, 2, 0, 11, 5, 9, 14, 12, 10, 4 },
    { 2, 8, 11, 13, 15, 7, 6, 14, 3, 1, 9, 4, 0, 10, 12, 5 }
};

static uint8_t qt1[2][16] =
{   { 14, 12, 11, 8, 1, 2, 3, 5, 15, 4, 10, 6, 7, 0, 9, 13 }, 
    { 1, 14, 2, 11, 4, 12, 3, 7, 6, 13, 10, 5, 15, 9, 0, 8 }
};

static uint8_t qt2[2][16] = 
{   { 11, 10, 5, 14, 6, 13, 9, 0, 12, 8, 15, 3, 2, 4, 7, 1 },
    { 4, 12, 7, 5, 1, 6, 9, 10, 0, 14, 13, 8, 2, 11, 3, 15 }
};

static uint8_t qt3[2][16] = 
{   { 13, 7, 15, 4, 1, 2, 6, 14, 9, 11, 3, 0, 8, 5, 12, 10 },
    { 11, 9, 5, 1, 12, 3, 13, 14, 6, 4, 7, 15, 2, 0, 8, 10 }
};
 
static uint8_t qp(const uint32_t n, const uint8_t x)
{   uint8_t  a0, a1, a2, a3, a4, b0, b1, b2, b3, b4;

    a0 = x >> 4; b0 = x & 15;
    a1 = a0 ^ b0; b1 = ror4[b0] ^ ashx[a0];
    a2 = qt0[n][a1]; b2 = qt1[n][b1];
    a3 = a2 ^ b2; b3 = ror4[b2] ^ ashx[a2];
    a4 = qt2[n][a3]; b4 = qt3[n][b3];
    return (b4 << 4) | a4;
};


static u4byte  qt_gen = 0;
static u1byte  q_tab[2][256];

#define q(n,x)  q_tab[n][x]

static void gen_qtab(void)
{   u4byte  i;

    for(i = 0; i < 256; ++i)
    {       
        q(0,i) = qp(0, (u1byte)i);
        q(1,i) = qp(1, (u1byte)i);
    }
};



static uint32_t  mt_gen = 0;
static uint32_t  m_tab[4][256];

static void gen_mtab(void)
{   uint32_t  i, f01, f5b, fef;
    
    for(i = 0; i < 256; ++i)
    {
        f01 = q(1,i); f5b = ffm_5b(f01); fef = ffm_ef(f01);
        m_tab[0][i] = f01 + (f5b << 8) + (fef << 16) + (fef << 24);
        m_tab[2][i] = f5b + (fef << 8) + (f01 << 16) + (fef << 24);

        f01 = q(0,i); f5b = ffm_5b(f01); fef = ffm_ef(f01);
        m_tab[1][i] = fef + (fef << 8) + (f5b << 16) + (f01 << 24);
        m_tab[3][i] = f5b + (f01 << 8) + (fef << 16) + (f5b << 24);
    }
};

#define mds(n,x)    m_tab[n][x]

static uint32_t h_fun(TwofishInstance *instance, const uint32_t x, const uint32_t key[])
{   uint32_t  b0, b1, b2, b3;


    b0 = extract_byte(x, 0); b1 = extract_byte(x, 1); b2 = extract_byte(x, 2); b3 = extract_byte(x, 3);

    switch(instance->k_len)
    {
    case 4: b0 = q(1, (u1byte) b0) ^ extract_byte(key[3],0);
            b1 = q(0, (u1byte) b1) ^ extract_byte(key[3],1);
            b2 = q(0, (u1byte) b2) ^ extract_byte(key[3],2);
            b3 = q(1, (u1byte) b3) ^ extract_byte(key[3],3);
    case 3: b0 = q(1, (u1byte) b0) ^ extract_byte(key[2],0);
            b1 = q(1, (u1byte) b1) ^ extract_byte(key[2],1);
            b2 = q(0, (u1byte) b2) ^ extract_byte(key[2],2);
            b3 = q(0, (u1byte) b3) ^ extract_byte(key[2],3);
    case 2: b0 = q(0, (u1byte) (q(0, (u1byte) b0) ^ extract_byte(key[1],0))) ^ extract_byte(key[0],0);
            b1 = q(0, (u1byte) (q(1, (u1byte) b1) ^ extract_byte(key[1],1))) ^ extract_byte(key[0],1);
            b2 = q(1, (u1byte) (q(0, (u1byte) b2) ^ extract_byte(key[1],2))) ^ extract_byte(key[0],2);
            b3 = q(1, (u1byte) (q(1, (u1byte) b3) ^ extract_byte(key[1],3))) ^ extract_byte(key[0],3);
    }

    return  mds(0, b0) ^ mds(1, b1) ^ mds(2, b2) ^ mds(3, b3);

};


static u1byte  sb[4][256];


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



#define g0_fun(x)   h_fun(instance, x, instance->s_key)
#define g1_fun(x)   h_fun(instance, rotl(x,8), instance->s_key)

static void gen_mk_tab(TwofishInstance *instance, uint32_t key[])
{   uint32_t  i;
    u1byte  by;

	uint32_t *mk_tab = instance->mk_tab;

    switch(instance->k_len)
    {
    case 2: for(i = 0; i < 256; ++i)
            {
                by = (u1byte)i;
                mk_tab[0 + 4*i] = mds(0, q20(by)); mk_tab[1 + 4*i] = mds(1, q21(by));
                mk_tab[2 + 4*i] = mds(2, q22(by)); mk_tab[3 + 4*i] = mds(3, q23(by));

            }
            break;
    
    case 3: for(i = 0; i < 256; ++i)
            {
                by = (u1byte)i;

                mk_tab[0 + 4*i] = mds(0, q30(by)); mk_tab[1 + 4*i] = mds(1, q31(by));
                mk_tab[2 + 4*i] = mds(2, q32(by)); mk_tab[3 + 4*i] = mds(3, q33(by));
            }
            break;
    
    case 4: for(i = 0; i < 256; ++i)
            {
                by = (u1byte)i;
                mk_tab[0 + 4*i] = mds(0, q40(by)); mk_tab[1 + 4*i] = mds(1, q41(by));
                mk_tab[2 + 4*i] = mds(2, q42(by)); mk_tab[3 + 4*i] = mds(3, q43(by));

            }
    }
};


#    define g0_fun(x) ( mk_tab[0 + 4*extract_byte(x,0)] ^ mk_tab[1 + 4*extract_byte(x,1)] \
                      ^ mk_tab[2 + 4*extract_byte(x,2)] ^ mk_tab[3 + 4*extract_byte(x,3)] )
#    define g1_fun(x) ( mk_tab[0 + 4*extract_byte(x,3)] ^ mk_tab[1 + 4*extract_byte(x,0)] \
                      ^ mk_tab[2 + 4*extract_byte(x,1)] ^ mk_tab[3 + 4*extract_byte(x,2)] )




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

static uint32_t mds_rem(uint32_t p0, uint32_t p1)
{   uint32_t  i, t, u;

    for(i = 0; i < 8; ++i)
    {
        t = p1 >> 24;   // get most significant coefficient
        
        p1 = (p1 << 8) | (p0 >> 24); p0 <<= 8;  // shift others up
            
        // multiply t by a (the primitive element - i.e. left shift)

        u = (t << 1); 
        
        if(t & 0x80)            // subtract modular polynomial on overflow
        
            u ^= G_MOD; 

        p1 ^= t ^ (u << 16);    // remove t * (a * x^2 + 1)  

        u ^= (t >> 1);          // form u = a * t + t / a = t * (a + 1 / a); 
        
        if(t & 0x01)            // add the modular polynomial on underflow
        
            u ^= G_MOD >> 1;

        p1 ^= (u << 24) | (u << 8); // remove t * (a + 1/a) * (x^3 + x)
    }

    return p1;
};

/* initialise the key schedule from the user supplied key   */

uint32_t *twofish_set_key(TwofishInstance *instance, const uint32_t in_key[], const uint32_t key_len)
{   uint32_t  i, a, b, me_key[4], mo_key[4];
	uint32_t *l_key, *s_key;

	l_key = instance->l_key;
	s_key = instance->s_key;

    if(!qt_gen)
    {
        gen_qtab(); qt_gen = 1;
    }


    if(!mt_gen)
    {
        gen_mtab(); mt_gen = 1;
    }


    instance->k_len = key_len / 64;   /* 2, 3 or 4 */

    for(i = 0; i < instance->k_len; ++i)
    {
        a = LE32(in_key[i + i]);     me_key[i] = a;
        b = LE32(in_key[i + i + 1]); mo_key[i] = b;
        s_key[instance->k_len - i - 1] = mds_rem(a, b);
    }

    for(i = 0; i < 40; i += 2)
    {
        a = 0x01010101 * i; b = a + 0x01010101;
        a = h_fun(instance, a, me_key);
        b = rotl(h_fun(instance, b, mo_key), 8);
        l_key[i] = a + b;
        l_key[i + 1] = rotl(a + 2 * b, 9);
    }

    gen_mk_tab(instance, s_key);


    return l_key;
};



#define f_rnd(i)                                                    \
    t1 = g1_fun(blk[1]); t0 = g0_fun(blk[0]);                       \
    blk[2] = rotr(blk[2] ^ (t0 + t1 + l_key[4 * (i) + 8]), 1);      \
    blk[3] = rotl(blk[3], 1) ^ (t0 + 2 * t1 + l_key[4 * (i) + 9]);  \
    t1 = g1_fun(blk[3]); t0 = g0_fun(blk[2]);                       \
    blk[0] = rotr(blk[0] ^ (t0 + t1 + l_key[4 * (i) + 10]), 1);     \
    blk[1] = rotl(blk[1], 1) ^ (t0 + 2 * t1 + l_key[4 * (i) + 11])

exception_t* twofish_encrypt_block(block128_u* block, key256_u* fullkey){



}

#define i_rnd(i)                                                        \
        t1 = g1_fun(blk[1]); t0 = g0_fun(blk[0]);                       \
        blk[2] = rotl(blk[2], 1) ^ (t0 + t1 + l_key[4 * (i) + 10]);     \
        blk[3] = rotr(blk[3] ^ (t0 + 2 * t1 + l_key[4 * (i) + 11]), 1); \
        t1 = g1_fun(blk[3]); t0 = g0_fun(blk[2]);                       \
        blk[0] = rotl(blk[0], 1) ^ (t0 + t1 + l_key[4 * (i) +  8]);     \
        blk[1] = rotr(blk[1] ^ (t0 + 2 * t1 + l_key[4 * (i) +  9]), 1)

exception_t* twofish_decrypt_block(block128_t* block, key256_u* fullkey){

}




exception_t* twofish(key256_t* user_key, block128_t* blocks, int block_count, enum mode mode, enum encryption encryption, size_t* buffer_size){
uint8_t* function_name = "twofish()";
	exception_t* exception;

	// Validate parameters.
	if ( user_key == NULL ) {
		return exception_throw("user_key was NULL.", function_name);
	}
	if ( blocks == NULL ) {
		return exception_throw("blocks was NULL.", function_name);
	}
	if ( buffer_size == NULL ) {
		return exception_throw("buffer_size was NULL.", function_name);
	}

	// Run the appropirate algorithm.
	switch(mode) {
	case CUDA:
		switch(encryption) {
		case DECRYPT:
			exception = twofish_cuda_decrypt(user_key, blocks, block_count, buffer_size);
			break;
		case ENCRYPT:
			exception = twofish_cuda_encrypt(user_key, blocks, block_count, buffer_size);
			break;
		default:
			return exception_throw("Unrecognized encryption parameter for CUDA.", function_name);
		}
		break;
	case PARALLEL:
		switch(encryption) {
		case DECRYPT:
			exception = twofish_parallel_decrypt(user_key, blocks, block_count);
			break;
		case ENCRYPT:
			exception = twofish_parallel_encrypt(user_key, blocks, block_count);
			break;
		default:
			return exception_throw("Unrecognized encryption parameter for parallel.", function_name);
		}
		break;
	case SERIAL:
		switch(encryption) {
		case DECRYPT:
			exception = twofish_serial_decrypt(user_key, blocks, block_count);
			break;
		case ENCRYPT:
			exception = twofish_serial_encrypt(user_key, blocks, block_count);
			break;
		default:
			return exception_throw("Unrecognized encryption parameter for serial.", function_name);
		}
		break;
	default:
		return exception_throw("Unknown mode.", function_name);
	}
	if ( exception != NULL ) {
		return exception_append(exception, function_name);
	}

	// Return success.
	return NULL;

}



exception_t* twofish_serial_decrypt(key256_t* user_key, block128_t* blocks, int block_count) {
	uint8_t* function_name = "twofish_serial_decrypt()";
	exception_t* exception;
	uint32_t* subkey;
	//uint32_t a, b, c, d, e;

	// Validate parameters.
	if ( user_key == NULL ) {
		return exception_throw("user_key was NULL.", function_name);
	}
	if ( blocks == NULL ) {
		return exception_throw("blocks was NULL.", function_name);
	}

	// Initialize the subkey.
	exception = twofish_init_subkey(user_key, &subkey);
	if ( exception != NULL ) {
		return exception_append(exception, function_name);
	}

	// Decrypt each block.
	for ( int i = 0; i < block_count; i++ ) {
		exception = twofish_decrypt_block(&(blocks[i]), subkey);
		if ( exception != NULL ) {
			return exception_append(exception, function_name);
		}
	}

	// Free the subkey.
	exception = twofish_free_subkey(subkey);
	if ( exception != NULL ) {
		return exception_append(exception, function_name);
	}

	// Return success.
	return NULL;
}


exception_t* twofish_serial_encrypt(key256_t* user_key, block128_t* blocks, int block_count) {
	uint8_t* function_name = "twofish_serial_encrypt()";
	exception_t* exception;
	uint32_t* subkey;

	// Validate parameters.
	if ( user_key == NULL ) {
		return exception_throw("user_key was NULL.", function_name);
	}
	if ( blocks == NULL ) {
		return exception_throw("blocks was NULL.", function_name);
	}

    
    // Add the key struct to the union so that twofish can do indexing tricks
   key256_u key;
   key.key = *user_key;
    
    
	// Encrypt each block.
	for ( int i = 0; i < block_count; i++ ) {
        //Add the block to the union so that we can index it with array stuff
        block128_u block;
        block.data = blocks[i];
    
		exception = twofish_encrypt_block(&block, &key);
		if ( exception != NULL ) {
			return exception_throw("blocks was NULL.", function_name);
		}
	}

	// Free the subkey.
	exception = twofish_free_subkey(subkey);
	if ( exception != NULL ) {
		return exception_append(exception, function_name);
	}

	// Return success.
	return NULL;
}




exception_t* twofish_init_subkey(key256_t* user_key, uint32_t** subkey) {
	uint8_t* function_name = "twofish_init_subkey()";
	const int PREKEY_SIZE = 140;
	exception_t* exception;
	uint32_t* genkey;
	uint32_t a, b, c, d, e;

	// Validate parameters.
	if ( user_key == NULL ) {
		return exception_throw("user_key was NULL.", function_name);
	}
	if ( subkey == NULL ) {
		return exception_throw("subkey was NULL.", function_name);
	}

	// Allocate space for genkey.
	genkey = (uint32_t*)malloc(sizeof(uint32_t) * PREKEY_SIZE);
	if ( genkey == NULL ) {
		return exception_throw("Unable to allocate genkey.", function_name);
	}

	// Assign user_key to the genkey; making sure to properly little-endianized the user key.
	for ( int i = 0; i < 8; i++ ) {
		uint32_t word;
		// Get the key value.
		exception = key256_get_word(user_key, i, &word);
		if ( exception != NULL ) {
			return exception_append(exception, function_name);
		}

		// Endianize the key value.
		word = mirror_bytes32(word);

		// Set the key value.
		genkey[i] = word;
	}

	// Generate the prekey by the following affine recurrence.
	genkey += 8;
	uint32_t t = genkey[-1];
	for ( int i = 0; i < 132; ++i ) {
		genkey[i] = t = rotl_fixed(genkey[i-8] ^ genkey[i-5] ^ genkey[i-3] ^ t ^ 0x9E3779B9 ^ i, 11);
	}
	genkey -= 20;

#define LK(r, a, b, c, d, e)    {\
	a = genkey[(8-r)*4 + 0];	     \
	b = genkey[(8-r)*4 + 1];	     \
	c = genkey[(8-r)*4 + 2];	     \
	d = genkey[(8-r)*4 + 3];}

#define SK(r, a, b, c, d, e)    {\
	genkey[(8-r)*4 + 4] = a;	     \
	genkey[(8-r)*4 + 5] = b;	     \
	genkey[(8-r)*4 + 6] = c;	     \
	genkey[(8-r)*4 + 7] = d;}    \

	// Generare the subkey using the prekey.
	for ( int i = 0; i < 4 ; i++ )
	{
		afterS2(LK); afterS2(S3); afterS3(SK);
		afterS1(LK); afterS1(S2); afterS2(SK);
		afterS0(LK); afterS0(S1); afterS1(SK);
		beforeS0(LK); beforeS0(S0); afterS0(SK);
		genkey += 8*4;
		afterS6(LK); afterS6(S7); afterS7(SK);
		afterS5(LK); afterS5(S6); afterS6(SK);
		afterS4(LK); afterS4(S5); afterS5(SK);
		afterS3(LK); afterS3(S4); afterS4(SK);
	}
	afterS2(LK); afterS2(S3); afterS3(SK);
	genkey -= 108;

	// Assign output parameter.
	(*subkey) = genkey;

	// Return success.
	return NULL;
}


exception_t* twofish_free_subkey(uint32_t* subkey) {
	uint8_t* function_name = "twofish_free_subkey()";

	// Validate parameters.
	if ( subkey == NULL ) {
		return exception_throw("subkey was NULL.", function_name);
	}

	// Free the subkey.
	// Note that the original key be freed, too.
	free(subkey-8);

	// Return success.
	return NULL;
}


exception_t* twofish_cuda_decrypt(key256_t* user_key, block128_t* blocks, int block_count, size_t* buffer_size) {
    return NULL;
}


/**	Encrypt the specified array of 128-bit blocks through the CUDA twofish algorithm.
 *	@return	NULL on success, exception_t* on failure.
 */
exception_t* twofish_cuda_encrypt(key256_t* user_key, block128_t* blocks, int block_count, size_t* buffer_size){
    return NULL;
}


/**	Private inner function to prevent linking errors because nvcc does not like external libraries.
 *	@return 0 on success, -1 on failure.
 */
int twofish_cuda_decrypt_cu(uint32_t* subkey, block128_t* blocks, int block_count, size_t* buffer_size){
    return NULL;

}



int twofish_cuda_encrypt_cu(uint32_t* subkey, block128_t* blocks, int block_count, size_t* buffer_size){
    return NULL;

}


/**	Decrypt the specified array of 128-bit blocks in parallel through the twofish encryption algorithm.
 *	@return NULL on success, exception_t* on failure.
 */
exception_t* twofish_parallel_decrypt(key256_t* user_key, block128_t* blocks, int block_count){
    return NULL;
}






/**	Encrypt the specified array of 128-bit blocks in parallel through the twofish encryption algorithm.
 *	@return NULL on success, exception_t* on failure.
 */
exception_t* twofish_parallel_encrypt(key256_t* user_key, block128_t* blocks, int block_count){
    return NULL;
}

