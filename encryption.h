#ifndef encryption_H
#define encryption_H


#include "ccc/ccc.h"


// The buffer size for the name of the encryption enumeration.
#define ENCRYPTION_NAME_LENGTH 15


/**	Enumeration representing whether to encrypt or decrypt the file.
 */
enum encryption {
	DECRYPT,
	ENCRYPT
};


/**	Returns the human-readable name of the specified encryption enumeration.
 *	@out	name: The human-readable name of the encryption enumeration.
		The buffer size should be at least equal to the corresponding macro definition.
 *	@return	NULL on success, exception_t* on failure.
 */
exception_t* encryption_get_name(enum encryption encryption, char* name);


#endif // encryption_H
