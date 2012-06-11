#ifndef encryption_H
#define encryption_H


#include "ccc/ccc.h"


//! The buffer size for the human-readable name of the encryption enumeration.
#define ENCRYPTION_NAME_LENGTH 15


/*!	\brief Enumeration representing whether to encrypt or decrypt the file.
 */
enum encryption {
	DECRYPT,
	ENCRYPT
};


/*!	\brief Returns the human-readable name of the specified encryption enumeration.
 * 	\warning The buffer size of name should be greater than or equal to ENCRYPTION_NAME_LENGTH.
 *
 *	\param[in]	encryption	The encryption enumeration to get the human-readable name of.
 *	\param[out]	name		The human-readable name of the encryption enumeration.
 *	\return	NULL on success, exception_t* on failure.
 */
exception_t* encryption_get_name(enum encryption encryption, char* name);


#endif // encryption_H
