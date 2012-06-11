#ifndef typedef_H
#define typedef_H


#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "algorithm.h"
#include "ccc/ccc.h"
#include "encryption.h"
#include "mode.h"


/*!	\brief Print a usage message to stdout and exit the program.
 *
 * 	\param[in]	message	The error message to print to the user.
 */
void print_usage(char* message);


/*!	\brief Parse the arguments and set their values appropriately.
 *
 * 	\param[in]	argv		The arguments to parse.
 *	\param[out]	data_count	The number of times to collect data for each subsection of the report.
 *	\return NULL on success, exception_t* exception.
 */
exception_t* arguments_parse(char* argv[], int* data_count);


/*!	\brief Parse the number of times to collect data for each subsection of the report.
 *
 * 	\param[in]	argument	The argument to parse for the value of data_count.
 *	\param[out]	data_count	The number of times to collect data for each subsection of the report.
 *	\return	NULL on success, exception_t* on failure.
 */
exception_t* arguments_parse_data_count(char* argument, int* data_count);


/*!	\brief Parse the input filepath to make sure the file exists and is not a directory.
 *	\note Admittedly this is not _technically_ parsing.
 *
 * 	\param[in]	argument	The argument to parse for the filepath of the input file.
 *	\return	NULL on success, exception_t* on failure.
 */
exception_t* arguments_parse_input_filepath(char* argument);


/*!	\brief Validate user-supplied arguments.
 *
 * 	\param[in]	argc	The number of arguments supplied by the user.
 * 	\param[in]	argv	The actual arguments supplied by the user.
 *	\return	NULL on success, exception_t* on failure.
 */
exception_t* arguments_validate(int argc, char* argv[]);


#endif // typedef_H
