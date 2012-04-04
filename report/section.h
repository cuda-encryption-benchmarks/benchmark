
#ifndef section_H
#define section_H


#include "subsection.h"
#include "../ccc/ccc.h"
#include "../typedef.h"


// The number of subsections in the section class.
// Note the naming convention. The prefix "SECTION" designates
// the structure that the #define is in, while the suffix
// "SUBSECTION_COUNT" denotes the name of the property.
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
	// The subsections of the report, altogether containing the different modes
	// the algorithm is run in.
	subsection_t subsections[SECTION_SUBSECTION_COUNT];
} section_t;


/**	Iniitialize the specified section_t with the specified algorithm.
 *	@return	NULL on success, exception_t* on failure.
 */
exception_t* section_init(section_t* section, enum algorithm algorithm);


/**	Uninitialize the specified section_t.
 *	@return	NULL on success, exception_t* on failure.
 */
exception_t* section_free(section_t* section);


/**	Write the specified section_t to the specified pre-opened and writable file.
 *	@return	NULL on success, exception_t* on failure.
 */
exception_t* section_write(section_t* section, FILE* file);


#endif // section_H
