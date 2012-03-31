#include "key.h"


exception_t* key256_get_word(key256_t* key, int index, uint32_t* word) {
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


exception_t* key256_set_word(key256_t* key, int index, uint32_t word) {
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

