#ifndef typedef_H
#define typedef_H


#include "algorithm.h"
#include "ccc/ccc.h"
#include "encryption.h"
#include "mode.h"


/**	Print a usage message to stdout and exit the program.
 */
void print_usage(char* message);


/**	Parse the arguments and set their values appropriately.
 *	@out	data_count: The number of times to collect data for each subsection of the report.
 *	@return NULL on success, exception_t* exception.
 */
exception_t* arguments_parse(char* argv[], int* data_count);


/**	Parse the number of times to collect data for each subsection of the report.
 *	@out	data_count: The number of times to collect data for each subsection of the report.
 *	@return	NULL on success, exception_t* on failure.
 */
exception_t* arguments_parse_data_count(char* argument, int* data_count);


/**	Validate user-supplied arguments.
 *	@return	NULL on success, exception_t* on failure.
 */
exception_t* arguments_validate(int argc, char* argv[]);


#endif // typedef_H
