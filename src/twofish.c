/* See "twofish.h" for legal information. */

#include "twofish.h"

// BEGIN Wall of Twofish statics.
uint8_t  tab_5b[4] = { 0, G_M >> 2, G_M >> 1, (G_M >> 1) ^ (G_M >> 2) };
uint8_t  tab_ef[4] = { 0, (G_M >> 1) ^ (G_M >> 2), G_M >> 1, G_M >> 2 };

uint8_t ror4[16] = { 0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15 };
uint8_t ashx[16] = { 0, 9, 2, 11, 4, 13, 6, 15, 8, 1, 10, 3, 12, 5, 14, 7 };

uint8_t qt0[2][16] = {
	{ 8, 1, 7, 13, 6, 15, 3, 2, 0, 11, 5, 9, 14, 12, 10, 4 },
	{ 2, 8, 11, 13, 15, 7, 6, 14, 3, 1, 9, 4, 0, 10, 12, 5 }
};

uint8_t qt1[2][16] = {
	{ 14, 12, 11, 8, 1, 2, 3, 5, 15, 4, 10, 6, 7, 0, 9, 13 },
	{ 1, 14, 2, 11, 4, 12, 3, 7, 6, 13, 10, 5, 15, 9, 0, 8 }
};

uint8_t qt2[2][16] = {
	{ 11, 10, 5, 14, 6, 13, 9, 0, 12, 8, 15, 3, 2, 4, 7, 1 },
	{ 4, 12, 7, 5, 1, 6, 9, 10, 0, 14, 13, 8, 2, 11, 3, 15 }
};

uint8_t qt3[2][16] = {
	{ 13, 7, 15, 4, 1, 2, 6, 14, 9, 11, 3, 0, 8, 5, 12, 10 },
	{ 11, 9, 5, 1, 12, 3, 13, 14, 6, 4, 7, 15, 2, 0, 8, 10 }
};

uint8_t twofish_qp(const uint32_t n, const uint8_t x) {
	uint8_t a0, a1, a2, a3, a4, b0, b1, b2, b3, b4;

	a0 = x >> 4;
	b0 = x & 15;
	a1 = a0 ^ b0;
	b1 = ror4[b0] ^ ashx[a0];
	a2 = qt0[n][a1];
	b2 = qt1[n][b1];
	a3 = a2 ^ b2;
	b3 = ror4[b2] ^ ashx[a2];
	a4 = qt2[n][a3];
	b4 = qt3[n][b3];
	return (b4 << 4) | a4;
}

uint8_t q_tab[2][256];
void twofish_gen_qtab(void) {
	uint32_t i;

	for(i = 0; i < 256; ++i) {
		q(0,i) = twofish_qp(0, (uint8_t)i);
		q(1,i) = twofish_qp(1, (uint8_t)i);
	}
}

uint32_t m_tab[4][256];
void twofish_gen_mtab(void) {
	uint32_t i, f01, f5b, fef;

	for(i = 0; i < 256; ++i) {
		f01 = q(1,i); f5b = ffm_5b(f01); fef = ffm_ef(f01);
		m_tab[0][i] = f01 + (f5b << 8) + (fef << 16) + (fef << 24);
		m_tab[2][i] = f5b + (fef << 8) + (f01 << 16) + (fef << 24);

		f01 = q(0,i); f5b = ffm_5b(f01); fef = ffm_ef(f01);
		m_tab[1][i] = fef + (fef << 8) + (f5b << 16) + (f01 << 24);
		m_tab[3][i] = f5b + (f01 << 8) + (fef << 16) + (f5b << 24);
	}
}

uint32_t twofish_mds_rem(uint32_t p0, uint32_t p1) {
	uint32_t  i, t, u;

	for(i = 0; i < 8; ++i) {
		t = p1 >> 24;   // get most significant coefficient

		p1 = (p1 << 8) | (p0 >> 24);
		p0 <<= 8;  // shift others up

		// multiply t by a (the primitive element - i.e. left shift)
		u = (t << 1);

		if(t & 0x80) {  // subtract modular polynomial on overflow
			u ^= G_MOD;
		}

		p1 ^= t ^ (u << 16);	// remove t * (a * x^2 + 1)  
		u ^= (t >> 1);	// form u = a * t + t / a = t * (a + 1 / a); 
		if(t & 0x01) {	 // add the modular polynomial on underflow
			u ^= G_MOD >> 1;
		}
		p1 ^= (u << 24) | (u << 8); // remove t * (a + 1/a) * (x^3 + x)
	}

	return p1;
}

uint32_t twofish_h_fun(twofish_instance_t* instance, const uint32_t x, const uint32_t key[]) {
	uint32_t  b0, b1, b2, b3;

	b0 = extract_byte(x, 0); b1 = extract_byte(x, 1); b2 = extract_byte(x, 2); b3 = extract_byte(x, 3);

	switch(instance->k_len) {
		// NOTE: We've harded the key length (k_len) to 4, but will leave in the case statements anyways...
		case 4: b0 = q(1, (uint8_t) b0) ^ extract_byte(key[3],0);
			b1 = q(0, (uint8_t) b1) ^ extract_byte(key[3],1);
			b2 = q(0, (uint8_t) b2) ^ extract_byte(key[3],2);
			b3 = q(1, (uint8_t) b3) ^ extract_byte(key[3],3);
		case 3: b0 = q(1, (uint8_t) b0) ^ extract_byte(key[2],0);
			b1 = q(1, (uint8_t) b1) ^ extract_byte(key[2],1);
			b2 = q(0, (uint8_t) b2) ^ extract_byte(key[2],2);
			b3 = q(0, (uint8_t) b3) ^ extract_byte(key[2],3);
		case 2: b0 = q(0, (uint8_t) (q(0, (uint8_t) b0) ^ extract_byte(key[1],0))) ^ extract_byte(key[0],0);
			b1 = q(0, (uint8_t) (q(1, (uint8_t) b1) ^ extract_byte(key[1],1))) ^ extract_byte(key[0],1);
			b2 = q(1, (uint8_t) (q(0, (uint8_t) b2) ^ extract_byte(key[1],2))) ^ extract_byte(key[0],2);
			b3 = q(1, (uint8_t) (q(1, (uint8_t) b3) ^ extract_byte(key[1],3))) ^ extract_byte(key[0],3);
	}

	return  mds(0, b0) ^ mds(1, b1) ^ mds(2, b2) ^ mds(3, b3);
}

void twofish_gen_mk_tab(twofish_instance_t* instance, uint32_t key[]) {
	uint32_t  i;
	uint8_t by;

	uint32_t *mk_tab = instance->mk_tab;

	switch(instance->k_len) {
	case 2:
		for(i = 0; i < 256; ++i) {
			by = (uint8_t)i;
			mk_tab[0 + 4*i] = mds(0, q20(by)); mk_tab[1 + 4*i] = mds(1, q21(by));
			mk_tab[2 + 4*i] = mds(2, q22(by)); mk_tab[3 + 4*i] = mds(3, q23(by));
		}
		break;
	case 3:
		for(i = 0; i < 256; ++i) {
			by = (uint8_t)i;
			mk_tab[0 + 4*i] = mds(0, q30(by)); mk_tab[1 + 4*i] = mds(1, q31(by));
			mk_tab[2 + 4*i] = mds(2, q32(by)); mk_tab[3 + 4*i] = mds(3, q33(by));
		}
		break;
	case 4:
		for(i = 0; i < 256; ++i) {
			by = (uint8_t)i;
			mk_tab[0 + 4*i] = mds(0, q40(by)); mk_tab[1 + 4*i] = mds(1, q41(by));
			mk_tab[2 + 4*i] = mds(2, q42(by)); mk_tab[3 + 4*i] = mds(3, q43(by));
		}
	}
}
// END Wall of Twofish statics.


exception_t* twofish(key256_t* user_key, block128_t* blocks, int block_count, enum mode mode, enum encryption encryption, size_t* buffer_size) {
	char* function_name = "twofish()";
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


exception_t* twofish_cuda_decrypt(key256_t* user_key, block128_t* blocks, int block_count, size_t* buffer_size) {
	char* function_name = "twofish_cuda_decrypt()";
	exception_t* exception;
	twofish_instance_t instance;

	// Validate parameters.
        #ifdef DEBUG_TWOFISH
        if ( user_key == NULL ) {
                return exception_throw("user_key was NULL.", function_name);
        }
        if ( blocks == NULL ) {
                return exception_throw("blocks was NULL.", function_name);
        }
        if ( block_count < 1 ) {
                return exception_throw("block_count less than 1.", function_name);
        }
        if ( buffer_size == NULL ) {
                return exception_throw("buffer_size was NULL.", function_name);
        }
        #endif

        // Initialize the Twofish instance.
        exception = twofish_instance_init(&instance, user_key);
        if ( exception != NULL ) {
                return exception_append(exception, function_name);
        }

        // Run the decryption algorithm from the .cu file.
        if ( twofish_cuda_decrypt_cu(&instance, blocks, block_count, buffer_size) == -1 ) {
                return exception_throw("CUDA encryption FAILED.", function_name);
        }

        // Return success.
        return NULL;
	
	
	return exception_throw("Not implemented.", function_name);
}


exception_t* twofish_cuda_encrypt(key256_t* user_key, block128_t* blocks, int block_count, size_t* buffer_size) {
	char* function_name = "twofish_cuda_encrypt()";
	exception_t* exception;
	twofish_instance_t instance;

	// Validate parameters.
	#ifdef DEBUG_TWOFISH
	if ( user_key == NULL ) {
		return exception_throw("user_key was NULL.", function_name);
	}
	if ( blocks == NULL ) {
		return exception_throw("blocks was NULL.", function_name);
	}
	if ( block_count < 1 ) {
		return exception_throw("block_count less than 1.", function_name);
	}
	if ( buffer_size == NULL ) {
		return exception_throw("buffer_size was NULL.", function_name);
	}
	#endif

	// Initialize the Twofish instance.
	exception = twofish_instance_init(&instance, user_key);
	if ( exception != NULL ) {
		return exception_append(exception, function_name);
	}

	// Run the encryption algorithm from the .cu file.
	if ( twofish_cuda_encrypt_cu(&instance, blocks, block_count, buffer_size) == -1 ) {
		return exception_throw("CUDA encryption FAILED.", function_name);
	}

	// Return success.
	return NULL;
}


void twofish_decrypt_block(block128_t* block, twofish_instance_t* instance) {
	uint32_t t0, t1, blk[4];

	uint32_t *l_key = instance->l_key;
	uint32_t *mk_tab = instance->mk_tab;

	// Input whitening.
	blk[0] = mirror_bytes32(block->x0) ^ l_key[4];
	blk[1] = mirror_bytes32(block->x1) ^ l_key[5];
	blk[2] = mirror_bytes32(block->x2) ^ l_key[6];
	blk[3] = mirror_bytes32(block->x3) ^ l_key[7];

	// Run the inverse 8 Fiestel network cycles.
	i_rnd(7); i_rnd(6); i_rnd(5); i_rnd(4);
	i_rnd(3); i_rnd(2); i_rnd(1); i_rnd(0);

	// Output whitening.
	block->x0 = mirror_bytes32(blk[2] ^ l_key[0]);
	block->x1 = mirror_bytes32(blk[3] ^ l_key[1]);
	block->x2 = mirror_bytes32(blk[0] ^ l_key[2]);
	block->x3 = mirror_bytes32(blk[1] ^ l_key[3]);
}


void twofish_encrypt_block(block128_t* block, twofish_instance_t* instance) {
	uint32_t t0, t1, blk[4];

	uint32_t* l_key = instance->l_key;
	uint32_t* mk_tab = instance->mk_tab;
	
	// Input whitening.
	blk[0] = mirror_bytes32(block->x0) ^ l_key[0];
	blk[1] = mirror_bytes32(block->x1) ^ l_key[1];
	blk[2] = mirror_bytes32(block->x2) ^ l_key[2];
	blk[3] = mirror_bytes32(block->x3) ^ l_key[3];

	// Run the 8 Fiestel network cycles.
	f_rnd(0); f_rnd(1); f_rnd(2); f_rnd(3);
	f_rnd(4); f_rnd(5); f_rnd(6); f_rnd(7);

	// Output whitening.
	block->x0 = mirror_bytes32(blk[2] ^ l_key[4]);
	block->x1 = mirror_bytes32(blk[3] ^ l_key[5]);
	block->x2 = mirror_bytes32(blk[0] ^ l_key[6]);
	block->x3 = mirror_bytes32(blk[1] ^ l_key[7]);
}


exception_t* twofish_instance_init(twofish_instance_t* instance, key256_t* user_key) {
	char* function_name = "twofish_instance_init()";
	exception_t* exception;
	uint32_t a, b, me_key[4], mo_key[4];
	uint32_t* l_key;
	uint32_t* s_key;
	int i;

	// Validate parameters.
	#ifdef DEBUG_TWOFISH
	if ( instance == NULL ) {
		return exception_throw("instance was NULL.", function_name);
	}
	if ( user_key == NULL ) {
		return exception_throw("user_key was NULL.", function_name);
	}
	#endif

	// Get pointers to some of the arrays in the key schedule.
	l_key = instance->l_key;
	s_key = instance->s_key;

	// Generate static Q and M tables.
	// NOTE: Traditional implementation will only generate tables once, but for benchmarking
	// purposes the tables are generated for each benchmark iteration.
	twofish_gen_qtab();
	twofish_gen_mtab();

	// Get the key length as a multiple of 64.
	//instance->k_len = key_len / 64;   /* 2, 3 or 4 */
	// NOTE: Hard coded to use a 256-bit key in this case.
	instance->k_len = 4;

	// Convert the bytes in 2k words of 32 bits each???
	for(i = 0; i < instance->k_len; ++i)
	{
		// Get the words from the user-supplied key.
		exception = key256_get_word(user_key, i + i, &a);
		if ( exception != NULL ) {
			exception_throw("Unable to get a.", function_name);
		}
		exception = key256_get_word(user_key, i + i + 1, &b);
		if ( exception != NULL ) {
			exception_throw("Unable to get b.", function_name);
		}

		// Covert to little-endian.
		a = mirror_bytes32(a);
		b = mirror_bytes32(b);

		// ???
		me_key[i] = a;
		mo_key[i] = b;
		s_key[instance->k_len - i - 1] = twofish_mds_rem(a, b);
	}

	// Not quite sure what this is supposed to represent...
	for(i = 0; i < 40; i += 2)
	{
		a = 0x01010101 * i;
		b = a + 0x01010101;
		a = twofish_h_fun(instance, a, me_key);
		b = rotl_fixed(twofish_h_fun(instance, b, mo_key), 8);
		l_key[i] = a + b;
		l_key[i + 1] = rotl_fixed(a + 2 * b, 9);
	}	

	// Generate the MK tables for the key schedule.
	twofish_gen_mk_tab(instance, s_key);

	// Return success.
	return NULL;
}


exception_t* twofish_instance_print(twofish_instance_t* instance) {
	#ifdef DEBUG_TWOFISH
	char* function_name = "twofish_instance_print()";
	#endif
	int i;
	int j;

	// Validate parameters (safety first).
	#ifdef DEBUG_TWOFISH
	if ( instance == NULL ) {
		return exception_throw("instance was NULL.", function_name);
	}
	#endif
	
	// l_key
	fprintf(stderr, "\n\nl_key:\n");
	for ( i = 0; i < 40; i += 5 ) {
		fprintf(stderr, "%8x %8x %8x %8x %8x\n", instance->l_key[i], instance->l_key[i+1], instance->l_key[i+2], instance->l_key[i+3], instance->l_key[i+4]);
	}

	// s_key
	fprintf(stderr, "s_key:\n");
	fprintf(stderr, "%8x %8x %8x %8x\n", instance->s_key[0], instance->s_key[1], instance->s_key[2], instance->s_key[3]);

	// mk_tab
	fprintf(stderr, "mk_tab:\n");
	for ( i = 0; i < 4; i++ ) {
		fprintf(stderr, "Row %i\n", i);
		for ( j = 0; j < 256; j += 8) {
			fprintf(stderr, "%8x %8x %8x %8x  %8x %8x %8x %8x\n", instance->mk_tab[i*256 + j], instance->mk_tab[i*256 + j+1], instance->mk_tab[i*256 + j+2], instance->mk_tab[i*256 + j+3], instance->mk_tab[i*256 + j+4], instance->mk_tab[i*256 + j+5], instance->mk_tab[i*256 + j+6], instance->mk_tab[i*256 + j+7]);
		}
	}

	// k_len
	fprintf(stderr, "k_len:\n%x\n\n", instance->k_len);

	// Return success.
	return NULL;
}


exception_t* twofish_parallel_decrypt(key256_t* user_key, block128_t* blocks, int block_count){
	char* function_name = "twofish_parallel_decrypt()";
	exception_t* exception;
	twofish_instance_t instance;

	// Validate parameters.
	#ifdef DEBUG_TWOFISH
	if ( user_key == NULL ) {
		return exception_throw("user_key was NULL.", function_name);
	}
	if ( blocks == NULL ) {
		return exception_throw("blocks was NULL.", function_name);
	}
	if ( block_count < 1 ) {
		return exception_throw("block_count less than 1.", function_name);
	}
	#endif

	// Initialize the Twofish instance.
	exception = twofish_instance_init(&instance, user_key);
	if ( exception != NULL ) {
		return exception_append(exception, function_name);
	}

	// Decrypt each block.
	#pragma omp parallel shared(blocks, instance, block_count)
	{
		int blocks_per_thread;
		int thread_count;
		int index;
		int i;

		#if defined (_OPENMP)
			thread_count = omp_get_num_threads();
			index = omp_get_thread_num();
		#endif

		// Decrypt the minimal number of blocks.
		blocks_per_thread = block_count / thread_count;
		for ( i = 0; i < blocks_per_thread; i++ ) {
			twofish_decrypt_block(&(blocks[index]), &instance);
			index += thread_count;
		}

		// Decrypt the extra blocks that fall outside the minimal number of blocks.
		if ( index < block_count ) {
			twofish_decrypt_block(&(blocks[index]), &instance);
		}
	}

	// Return success.
	return NULL;
}


exception_t* twofish_parallel_encrypt(key256_t* user_key, block128_t* blocks, int block_count){
	char* function_name = "twofish_parallel_encrypt()";
	exception_t* exception;
	twofish_instance_t instance;

	// Validate parameters.
	#ifdef DEBUG_TWOFISH
	if ( user_key == NULL ) {
		return exception_throw("user_key was NULL.", function_name);
	}
	if ( blocks == NULL ) {
		return exception_throw("blocks was NULL.", function_name);
	}
	if ( block_count < 1 ) {
		return exception_throw("block_count less than 1.", function_name);
	}
	#endif

	// Initialize the Twofish instance.
	exception = twofish_instance_init(&instance, user_key);
	if ( exception != NULL ) {
		return exception_append(exception, function_name);
	}

	// Encrypt each block.
	#pragma omp parallel shared(blocks, instance, block_count)
	{
		int blocks_per_thread;
		int thread_count;
		int index; 
		int i;

		#if defined (_OPENMP)
			thread_count = omp_get_num_threads();
			index = omp_get_thread_num();
		#endif

		// Encrypt the minimal number of blocks.
		blocks_per_thread = block_count / thread_count;
		for ( i = 0; i < blocks_per_thread; i++ ) {
			twofish_encrypt_block(&(blocks[index]), &instance);
			index += thread_count;
		}

		// Encrypt the extra blocks that fall outside the minimal number of blocks.
		if ( index < block_count ) {
			twofish_encrypt_block(&(blocks[index]), &instance);
		}

	}
	
	// Return success.
	return NULL;
}


exception_t* twofish_serial_decrypt(key256_t* user_key, block128_t* blocks, int block_count) {
	char* function_name = "twofish_serial_decrypt()";
	exception_t* exception;
	twofish_instance_t instance;
	int i;

	// Validate parameters.
	#ifdef DEBUG_TWOFISH
	if ( user_key == NULL ) {
		return exception_throw("user_key was NULL.", function_name);
	}
	if ( blocks == NULL ) {
		return exception_throw("blocks was NULL.", function_name);
	}
	#endif

	// Initialize the Twofish instance.
	exception = twofish_instance_init(&instance, user_key);
	if ( exception != NULL ) {
		return exception_append(exception, function_name);
	}

	// Decrypt each block.
	for ( i = 0; i < block_count; i++ ) {
		twofish_decrypt_block(&(blocks[i]), &instance);
	}

	// Return success.
	return NULL;
}


exception_t* twofish_serial_encrypt(key256_t* user_key, block128_t* blocks, int block_count) {
	char* function_name = "twofish_serial_encrypt()";
	exception_t* exception;
	twofish_instance_t instance;
	int i;

	// Validate parameters.
	#ifdef DEBUG_TWOFISH
	if ( user_key == NULL ) {
		return exception_throw("user_key was NULL.", function_name);
	}
	if ( blocks == NULL ) {
		return exception_throw("blocks was NULL.", function_name);
	}
	#endif

	// Initialize the Twofish instance.
	exception = twofish_instance_init(&instance, user_key);
	if ( exception != NULL ) {
		return exception_append(exception, function_name);
	}

	// Encrypt each block.
	for ( i = 0; i < block_count; i++ ) {
		twofish_encrypt_block(&(blocks[i]), &instance);
	}

	// Return success.
	return NULL;
}

