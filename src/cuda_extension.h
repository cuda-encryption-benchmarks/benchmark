#ifndef cuda_extension_H
#define cuda_extension_H


#include <stdio.h>
#include <inttypes.h>

#include "block128.h"


/*!	\brief Gets the number of CUDA-capable devices.
 *
 * 	\param[out]	device_count	The number of CUDA-capable devices on the host.
 *	\return	0 on success, -1 on failure.
 */
#ifdef __cplusplus
extern "C"
#endif
int cuda_device_count(int* device_count);


/*!	\brief Private function of report_write_system_information_cuda_devices() that writes
 * 		the specified CUDA device in the report's expected LaTeX format. 
 * 	\warning This function is highly coupled with its parent function and should not 
 * 		be used for anything else.
 * 	\note This function was moved outside of the report file because it uses the CUDA API.
 *
 *	\return	0 on success, -1 on failure.
 */
#ifdef __cplusplus
extern "C"
#endif
int cuda_device_properties_report_write(FILE* file, int device_number);


/*!	\brief Helper function that determines the maximum number of threads and blocks that
 * 		the specified device may run given the specified register usage.
 * 	\note This function probably doesn't do what I want it to do, but a register_per_thread
 * 		count of 8 is probably a safe bet for the maximum possible number of threads.
 *
 * 	\param[in]	device_number		Which CUDA device to get the data from.
 * 	\param[in]	registers_per_thread	The number of registers each thread uses when executing on the GPU.
 *	\param[out]	thread_count		The maximum number of threads to launch with.
 *	\param[out]	block_count		The maximum number of blocks to launch with.
 *	\return	0 on success, -1 on failure.
 */
int cuda_get_block_and_thread_count_max(int device_number, int registers_per_thread, int* block_count, int* thread_count);


#endif
