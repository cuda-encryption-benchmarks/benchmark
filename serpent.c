#include "serpent.h"


void sbox0_encrypt(uint32* r0, uint32* r1, uint32* r2, uint32* r3, uint32* r4) {
        *r3 ^= *r0;
        *r4 = *r1;
        *r1 &= *r3;
        *r4 ^= *r2;
        *r1 ^= *r0;
        *r0 |= *r3;
        *r0 ^= *r4;
        *r4 ^= *r3;
        *r3 ^= *r2;
        *r2 |= *r1;
        *r2 ^= *r4;
        *r4 = ~*r4;
        *r4 |= *r1;
        *r1 ^= *r3;
        *r1 ^= *r4;
        *r3 |= *r0;
        *r1 ^= *r3;
        *r4 ^= *r3;
}


void sbox1_encrypt(uint32* r0, uint32* r1, uint32* r2, uint32* r3, uint32* r4) {
    *r0 = ~*r0;
    *r2 = ~*r2;
    *r4 = *r0;
    *r0 &= *r1;
    *r2 ^= *r0;
    *r0 |= *r3;
    *r3 ^= *r2;
    *r1 ^= *r0;
    *r0 ^= *r4;
    *r4 |= *r1;
    *r1 ^= *r3;
    *r2 |= *r0;
    *r2 &= *r4;
    *r0 ^= *r1;
    *r1 &= *r2;
    *r1 ^= *r0;
    *r0 &= *r2;
    *r0 ^= *r4;
}


void sbox2_encrypt(uint32* r0, uint32* r1, uint32* r2, uint32* r3, uint32* r4) {
        *r4 = *r0;
        *r0 &= *r2;
        *r0 ^= *r3;
        *r2 ^= *r1;
        *r2 ^= *r0;
        *r3 |= *r4;
        *r3 ^= *r1;
        *r4 ^= *r2;
        *r1 = *r3;
        *r3 |= *r4;
        *r3 ^= *r0;
        *r0 &= *r1;
        *r4 ^= *r0;
        *r1 ^= *r3;
        *r1 ^= *r4;
        *r4 = ~*r4;
}


void sbox3_encrypt(uint32* r0, uint32* r1, uint32* r2, uint32* r3, uint32* r4) {
        *r4 = *r0;
        *r0 |= *r3;
        *r3 ^= *r1;
        *r1 &= *r4;
        *r4 ^= *r2;
        *r2 ^= *r3;
        *r3 &= *r0;
        *r4 |= *r1;
        *r3 ^= *r4;
        *r0 ^= *r1;
        *r4 &= *r0;
        *r1 ^= *r3;
        *r4 ^= *r2;
        *r1 |= *r0;
        *r1 ^= *r2;
        *r0 ^= *r3;
        *r2 = *r1;
        *r1 |= *r3;
        *r1 ^= *r0;
}


void sbox4_encrypt(uint32* r0, uint32* r1, uint32* r2, uint32* r3, uint32* r4) {
        *r1 ^= *r3;
        *r3 = ~*r3;
        *r2 ^= *r3;
        *r3 ^= *r0;
        *r4 = *r1;
        *r1 &= *r3;
        *r1 ^= *r2;
        *r4 ^= *r3;
        *r0 ^= *r4;
        *r2 &= *r4;
        *r2 ^= *r0;
        *r0 &= *r1;
        *r3 ^= *r0;
        *r4 |= *r1;
        *r4 ^= *r0;
        *r0 |= *r3;
        *r0 ^= *r2;
        *r2 &= *r3;
        *r0 = ~*r0;
        *r4 ^= *r2;
}


void sbox5_encrypt(uint32* r0, uint32* r1, uint32* r2, uint32* r3, uint32* r4) {
        *r0 ^= *r1;
        *r1 ^= *r3;
        *r3 = ~*r3;
        *r4 = *r1;
        *r1 &= *r0;
        *r2 ^= *r3;
        *r1 ^= *r2;
        *r2 |= *r4;
        *r4 ^= *r3;
        *r3 &= *r1;
        *r3 ^= *r0;
        *r4 ^= *r1;
        *r4 ^= *r2;
        *r2 ^= *r0;
        *r0 &= *r3;
        *r2 = ~*r2;
        *r0 ^= *r4;
        *r4 |= *r3;
        *r2 ^= *r4;
}


void sbox6_encrypt(uint32* r0, uint32* r1, uint32* r2, uint32* r3, uint32* r4) {
        *r2 = ~*r2;
        *r4 = *r3;
        *r3 &= *r0;
        *r0 ^= *r4;
        *r3 ^= *r2;
        *r2 |= *r4;
        *r1 ^= *r3;
        *r2 ^= *r0;
        *r0 |= *r1;
        *r2 ^= *r1;
        *r4 ^= *r0;
        *r0 |= *r3;
        *r0 ^= *r2;
        *r4 ^= *r3;
        *r4 ^= *r0;
        *r3 = ~*r3;
        *r2 &= *r4;
        *r2 ^= *r3;
}


void sbox7_encrypt(uint32* r0, uint32* r1, uint32* r2, uint32* r3, uint32* r4) {
        *r4 = *r2;
        *r2 &= *r1;
        *r2 ^= *r3;
        *r3 &= *r1;
        *r4 ^= *r2;
        *r2 ^= *r1;
        *r1 ^= *r0;
        *r0 |= *r4;
        *r0 ^= *r2;
        *r3 ^= *r1;
        *r2 ^= *r3;
        *r3 &= *r0;
        *r3 ^= *r4;
        *r4 ^= *r2;
        *r2 &= *r0;
        *r4 = ~*r4;
        *r2 ^= *r4;
        *r4 &= *r0;
        *r1 ^= *r3;
        *r4 ^= *r1;
}


exception_t* serpent_encrypt_serial(serpent_key* user_key, block128* blocks, int block_count) {
	char* function_name = "serpent_encrypt_serial()";
	exception_t* exception;
	uint32* subkey;

	// Generate the key.
	exception = serpent_init_key(user_key, &subkey);
	if ( exception != NULL ) {
		return exception_append(exception, function_name);
	}

	// Encrypt each block.
	for ( int i = 0; i < block_count; i++ ) {
		// Apply first 31 S-boxes.
			// Key mixing.
			// S-boxes.
			// Linear transformation.

		// Apply final S-box.
	}

	return exception_throw("Function not implemented.", function_name);
}


exception_t* serpent_init_key(serpent_key* user_key, uint32** subkey) {
	char* function_name = "serpent_init_key()";
	const int PREKEY_SIZE = 132;
	exception_t* exception;
	uint32* prekey;

	// Validate parameters.
	if ( user_key == NULL ) {
		return exception_throw("user_key was NULL.", function_name);
	}
	if ( subkey == NULL ) {
		return exception_throw("subkey was NULL.", function_name);
	}

	// Little endianize the key.
	for ( int i = 0; i < 8; i++ ) {
		uint32 word;
		// Get the key value.
		exception = serpent_key_get_word(user_key, i, &word);
		if ( exception != NULL ) {
			return exception_append(exception, function_name);
		}

		// Endianize the key value.
		exception = mirror_bytes32(word, &word);
		if ( exception != NULL ) {
			return exception_append(exception, function_name);
		}

		// Set the key value.
		exception = serpent_key_set_word(user_key, i, word);
		if ( exception != NULL ) {
			return exception_append(exception, function_name);
		}
	}

	// Allocate space for prekey.
	prekey = (uint32*)malloc(sizeof(uint32) * PREKEY_SIZE);
	if ( prekey == NULL ) {
		return exception_throw("Unable to allocate prekey.", function_name);
	}

	// Generate prekey by the following affine recurrence.
	prekey += 8;
	uint32 t = prekey[-1];
	for ( int i = 0; i < 132; ++i ) {
		prekey[i] = t = rotlFixed(prekey[i-8] ^ prekey[i-5] ^ prekey[i-3] ^ t ^ 0x9E3779B9 ^ i, 11);
	}
	//prekey -= 20;

	printf("Phrekee:\n");
	for ( int i = 0; i < 132; i++ ) {
		if ( i % 4 == 0 ) {
			printf("\n");
		}
		printf("%X ", prekey[i]);
	}

	//printf("Keymabee:\n\n%X %X %X %X\n%X %X %X %X\n\n", user_key->key0.x0, user_key->key0.x1, user_key->key0.x2, user_key->key0.x3, user_key->key1.x0, user_key->key1.x1, user_key->key1.x2, user_key->key1.x3);

	return exception_throw("Function not implemented.", function_name);
}


exception_t* serpent_free_key(serpent_key* user_key, uint32* subkey) {
	char* function_name = "serpent_free_key()";

	// Validate parameters.
	if ( subkey == NULL ) {
		return exception_throw("subkey was NULL.", function_name);
	}

	// Free the subkey.
	free(subkey);

	// Return success.
	return NULL;
}


exception_t* serpent_key_get_word(serpent_key* key, int index, uint32* word) {
	char* function_name = "serpent_key_get_word()";

	// Validate parameters.
	if ( key == NULL ) {
		return exception_throw("key was NULL.", function_name);
	}
	if ( word == NULL ) {
		return exception_throw("word was NULL.", function_name);
	}
	if ( index < 0 || index > 7 ) {
		return exception_throw("Invalid index.", function_name);
	}

	// Set output parameter.
	switch(index) {
	case 0:
		(*word) = key->key0.x0;
		break;
	case 1:
		(*word) = key->key0.x1;
		break;
	case 2:
		(*word) = key->key0.x2;
		break;
	case 3:
		(*word) = key->key0.x3;
		break;
	case 4:
		(*word) = key->key1.x0;
		break;
	case 5:
		(*word) = key->key1.x1;
		break;
	case 6:
		(*word) = key->key1.x2;
		break;
	case 7:
		(*word) = key->key1.x3;
		break;
	default:
		return exception_throw("Unknown index.", function_name);
	}

	// Return success.
	return NULL;
}


exception_t* serpent_key_set_word(serpent_key* key, int index, uint32 word) {
	char* function_name = "serpent_key_set_word()";

	// Validate parameters.
	if ( key == NULL ) {
		return exception_throw("key was NULL.", function_name);
	}
	if ( index < 0 || index > 7 ) {
		return exception_throw("Invalid index.", function_name);
	}

	// Set output parameter.
	switch(index) {
	case 0:
		key->key0.x0 = word;
		break;
	case 1:
		key->key0.x1 = word;
		break;
	case 2:
		key->key0.x2 = word;
		break;
	case 3:
		key->key0.x3 = word;
		break;
	case 4:
		key->key1.x0 = word;
		break;
	case 5:
		key->key1.x1 = word;
		break;
	case 6:
		key->key1.x2 = word;
		break;
	case 7:
		key->key1.x3 = word;
		break;
	default:
		return exception_throw("Unknown index.", function_name);
	}

	// Return success.
	return NULL;
}
