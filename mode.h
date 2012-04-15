#ifndef mode_H
#define mode_H


// Buffer size for getting the mode name.
// NOTE: May not be implemented in all calls currently.
#define MODE_NAME_LENGTH 15


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
 *		Buffer length should be defined using the macro.
 *	@return	NULL on success, exception_t* on failure.
 */
exception_t* mode_get_name(enum mode mode, char* name);


#endif // mode_H
