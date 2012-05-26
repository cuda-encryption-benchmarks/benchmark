#ifndef cuda_extension_H
#define cuda_extension_H


#include <stdio.h>

#include "block128.h"


/**	Gets the number of CUDA-capable devices.
 *	@return	0 on success, -1 on failure.
 */
#ifdef __cplusplus
extern "C"
#endif
int cuda_device_count(int* device_count);


/**	Private function of report_write_system_information_cuda_devices() that writes the specified CUDA device in the report's expected LaTeX format.
 *	This function is highly coupled with its parent function and should not be used for anything else. Ever. This function had to be moved outside
 *	of the report file due to the way CUDA compilation works.
 *	@return	0 on success, -1 on failure.
 */
#ifdef __cplusplus
extern "C"
#endif
int cuda_device_properties_report_write(FILE* file, int device_number);


/**	Helper function that determines the maximum number of threads and blocks the specified device may run given the specified register usage.
 *	@out	thread_count: The maximum number of threads to launch with.
 *	@out	block_count: The maximum number of blocks to launch with.
 *	@return	0 on success, -1 on failure.
 */
int cuda_get_block_and_thread_count_max(int device_number, int registers_per_thread, int* block_count, int* thread_count);


#endif
