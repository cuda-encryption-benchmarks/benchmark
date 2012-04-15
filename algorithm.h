#ifndef algorithm_H
#define algorithm_H


// Buffer size for algorithm_get_name function.
// NOTE: May not be used in all calls currently.
#define ALGORITHM_NAME_LENGTH 50


#include "ccc/ccc.h"


/**	Enumeration representing the different encryption algorithms.
 */
enum algorithm {
	AES,
	SERPENT,
	TWOFISH
};


/**	Get the human-readable name of the specified algorithm.
 *	@out	name: The human-readable name of the algorithm. Use macro for buffer length.
 *	@return	NULL on success, exception_t* on failure.
 */
exception_t* algorithm_get_name(enum algorithm algorithm, char* name);


#endif // algorithm_H
