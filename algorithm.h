#ifndef algorithm_H
#define algorithm_H


/*! \brief Buffer size for the algorithm_get_name() function.
 *   NOTE: May not currently be used in all calls; this behaviour
 *   is bad and should be fixed when noticed.
 */
#define ALGORITHM_NAME_LENGTH 50


#include "ccc/ccc.h"


/*!	\brief	Enumeration representing the different encryption algorithms.
 */
enum algorithm {
	AES,
	SERPENT,
	TWOFISH
};


/*!	\brief	Returns the human-readable name of the specified algorithm.
 *
 * 	\param[in]	algorithm	The algorithm to get the human-readable name of.
 *	\param[out]	name		The human-readable name of the algorithm. Use macro for buffer length.
 *	\return	NULL on success, exception_t* on failure.
 */
exception_t* algorithm_get_name(enum algorithm algorithm, char* name);


#endif // algorithm_H
