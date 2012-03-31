#ifndef mirror_bytes_H
#define mirror_bytes_H


#include "ccc/ccc.h"
#include "block128.h"


/**	Mirrors the bytes in the specified uint32_t.
 *	@return	A uint32_t with the bytes mirroted.
 */
uint32_t mirror_bytes32(uint32_t x);


#endif
