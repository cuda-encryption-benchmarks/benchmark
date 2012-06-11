#ifndef mode_H
#define mode_H


/*! \brief Buffer size for getting the mode name.
 *  \note May not be implemented in all calls currently. Should be implemented
 *  if noticed.
 */
#define MODE_NAME_LENGTH 15


#include "ccc/ccc.h"


/*!	\brief Enumeration representing how to run the algorithm.
 */
enum mode {
	CUDA,
	PARALLEL,
	SERIAL
};


/*!	\brief Get the human-readable name of the specified mode.
 * 	\warning Buffer length should be at least as long as MODE_NAME_LENGTH.
 *
 * 	\param[in]	mode	The mode to get the human-readable name from.
 *	\param[out]	name	The human-readable representation of the specified mode.
 *	\return	NULL on success, exception_t* on failure.
 */
exception_t* mode_get_name(enum mode mode, char* name);


#endif // mode_H
