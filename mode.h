#ifndef mode_H
#define mode_H


#include "ccc/ccc.h"


/**	Enumeration representing how to run the algorithm.
 */
enum mode {
	CUDA,
	PARALLEL,
	SERIAL
};


/**	Get the function name of the specified mode.
 *	@out	name: The human-readable representation of the specified mode.
 *		Buffer length should be at least 50.
 *	@return	NULL on success, exception_t* on failure.
 */
exception_t* mode_get_name(enum mode mode, char* name);


#endif // mode_H
