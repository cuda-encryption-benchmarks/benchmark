#ifndef typedef_H
#define typedef_H


#include "ccc/ccc.h"


/**	Enumeration representing the different encryption algorithms.
 */
enum algorithm {
	AES,
	SERPENT,
	TWOFISH
};


/**	Enumeration representing whether to encrypt or decrypt the file.
 */
enum encryption {
	DECRYPT,
	ENCRYPT
};


/**	Enumeration representing how to run the algorithm.
 */
enum mode {
	CUDA,
	PARALLEL,
	SERIAL
};


/**	Print a usage message to stdout and exit the program.
 */
void print_usage(char* message);


/**	Parse the arguments and set their values appropriately.
 *	@out	algorithm: The algorithm to encrypt/decrypt with.
 *	@out	mode: The methodology for running the cipher.
 *	@out	encryption: Wether to encryption/decrypt the file.
 *	@return NULL on success, exception_t* exception.
 */
exception_t* arguments_parse(char* argv[], enum algorithm* algorithm, enum mode* mode, enum encryption* encryption);


/**	Parse the specified argument for the encryption algorithm.
 *	@out	algorithm: The encryption/decryption algorithm to use.
 *	@return	NULL on success, exception_t* on failure.
 */
exception_t* arguments_parse_algorithm(char* argument, enum algorithm* algorithm);


/**	Parse the specified argument for the encryption enumeration.
 *	@out	encryption: The encryption/decryption method to use.
 *	@return	NULL on success, exception_t* on failure.
 */
exception_t* arguments_parse_encryption(char* argument, enum encryption* encryption);


/**	Parse the specified argument for the mode to run the cipher in.
 *	@out	mode: The methodology of how to run the cipher.
 *	@return	NULL on success, exception_t* on failure.
 */
exception_t* arguments_parse_mode(char* argument, enum mode* mode);


/**	Validate user-supplied arguments.
 *	@return	NULL on success, exception_t* on failure.
 */
exception_t* arguments_validate(int argc, char* argv[]);


#endif // typedef_H
