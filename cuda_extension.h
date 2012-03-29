#ifndef cuda_extension_H
#define cuda_extension_H


#include <stdio.h>


/**	Helper function that determines the maximum number of threads and blocks the specified device may run given the specified register usage.
 *	@out	thread_count: The maximum number of threads to launch with.
 *	@out	block_count: The maximum number of blocks to launch with.
 *	@return	0 on success, -1 on failure.
 */
int cuda_get_block_and_thread_count_max(int device_number, int registers_per_thread, int* block_count, int* thread_count);


#endif
