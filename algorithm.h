#ifndef algorithm_H
#define algorithm_H


#include "ccc/ccc.h"


/**	Enumeration representing the different encryption algorithms.
 */
enum algorithm {
	AES,
	SERPENT,
	TWOFISH
};


/**	Get the human-readable name of the specified algorithm.
 *	@out	name: The human-readable name of the algorithm. Buffer length should be at least 50.
 *	@return	NULL on success, exception_t* on failure.
 */
exception_t* algorithm_get_name(enum algorithm algorithm, char* name);


#endif // algorithm_H
