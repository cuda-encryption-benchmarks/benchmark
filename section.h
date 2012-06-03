
#ifndef section_H
#define section_H


#include "subsection.h"
#include "algorithm.h"
#include "ccc/ccc.h"
#include "key.h"
#include "typedef.h"


// The number of subsections in the section class.
// Note the naming convention. The prefix "SECTION" designates
// the structure that the #define is in, while the suffix
// "SUBSECTION_*" denotes the name of the property.
// It's really quite logical once your head stops hurting.
#define SECTION_SUBSECTION_COUNT 3
// The section for CUDA data.
#define SECTION_SUBSECTION_CUDA 2
// The section for parallel data.
#define SECTION_SUBSECTION_PARALLEL 1
// The section for serial data.
#define SECTION_SUBSECTION_SERIAL 0


// An abstraction which represents a section of the report.
// In this case, a section of the report is represented by
// a single encryption algorithm. Since an encryption algorithm
// has both encryption and decryption parts, each section must
// likewise have two parts to its data.
typedef struct {
	// The algorithm run by the specified section.
	enum algorithm algorithm;
	// The key to use for the specified algorithm.
	key256_t key;
	// The subsections of the report, altogether containing the different modes
	// the algorithm is run in.
	subsection_t subsections[SECTION_SUBSECTION_COUNT];
} section_t;


/**	Executes the data-gathering phase for the specified section.
 *	@return	NULL on success, exception_t* on failure.
 */
exception_t* section_execute(section_t* section, char* input_filepath);


/**	Iniitialize the specified section_t with the specified algorithm.
 *	@return	NULL on success, exception_t* on failure.
 */
exception_t* section_init(section_t* section, int data_count, enum algorithm algorithm);


/**	Uninitialize the specified section_t.
 *	@return	NULL on success, exception_t* on failure.
 */
exception_t* section_free(section_t* section);


/**	Write the appendix data of specified section_t to the specified pre-opened and writable file.
 *	@return	NULL on success, exception_t* on failure.
 */
exception_t* section_write_appendix(section_t* section, FILE* file);


/**	Write the results of the specified section_t* to the specified file.
 *	@return	NULL on success, exception_t* on failure.
 */
exception_t* section_write_results_summary(section_t* section, FILE* file, off_t size);


/**	Write the gain results of the speicfied section_t* to the specified file.
 *	@return	NULL on success, exception_t* on failure.
 */
exception_t* section_write_results_gain(section_t* section, FILE* file, off_t size);


/**	Write the results table for either the encrypt or decrypt version from the specified section.
 *	@return	NULL on success, exception-t* on failure.
 */
exception_t* section_write_results_summary_table(section_t* section, FILE* file, off_t size, enum encryption encryption);


#endif // section_H
