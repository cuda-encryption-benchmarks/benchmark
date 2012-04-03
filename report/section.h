
#ifndef section_H
#define section_H


#include "benchmark_data.h"
#include "../ccc/ccc.h"
#include "../typedef.h"


// The number of runs to collect data for in a section.
#define SECTION_ITERATION_COUNT 10


// An abstraction which represents a section of the report.
// In this case, a section of the report is represented by
// a single encryption algorithm. Since an encryption algorithm
// has both encryption and decryption parts, each section must
// likewise have two parts to its data.
typedef struct {
	// The algorithm run by the specified section.
	enum algorithm algorithm;
	// The mode that the algorithm is run in.
	enum mode mode;
	// The data gathered during encryption.
	benchmark_data_t data_encrypt[SECTION_ITERATION_COUNT];
	// The data gathered during decryption.
	benchmark_data_t data_decrypt[SECTION_ITERATION_COUNT];
} section_t;


/**	Iniitialize the specified section_t with the specified algorithm and encryption mode.
 *	@return	NULL on success, exception_t* on failure.
 */
exception_t* section_init(section_t* section, enum algorithm algorithm, enum mode mode);


/**	Uninitialize the specified section_t.
 *	@return	NULL on success, exception_t* on failure.
 */
exception_t* section_free(section_t* section);


/**	Write the specified section_t to the specified pre-opened and writable file.
 *	@return	NULL on success, exception_t* on failure.
 */
exception_t* section_write(section_t* section, FILE* file);


#endif // section_H
