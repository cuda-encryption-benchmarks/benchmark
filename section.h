
#ifndef section_H
#define section_H


#include "subsection.h"
#include "algorithm.h"
#include "ccc/ccc.h"
#include "key.h"
#include "typedef.h"


/*! \brief The number of subsections in the section class.
 *  Note the naming convention. The prefix "SECTION" designates
 *  the structure that the \#define is in, while the suffix
 *  "SUBSECTION_*" denotes the name of the property.
 *  It's really quite logical once your head stops hurting.
 */
#define SECTION_SUBSECTION_COUNT 3
//! The section for CUDA data.
#define SECTION_SUBSECTION_CUDA 2
//! The section for parallel data.
#define SECTION_SUBSECTION_PARALLEL 1
//! The section for serial data.
#define SECTION_SUBSECTION_SERIAL 0


/*! \brief An abstraction which represents a section of the report.
 *  	In this case, a section of the report is represented by
 *  	a single encryption algorithm. Since an encryption algorithm
 *  	has both encryption and decryption parts, each section must
 *  	likewise have two parts to its data.
 *  \note In retrospect, it would have made more sense to treat
 *  	the encryption and decryption algorithms as separate algorithms
 *  	and given both encryption and decryption their own sections.
 */
typedef struct {
	//! The algorithm run by the section.
	enum algorithm algorithm;
	//! The key to use for the specified algorithm.
	key256_t key;
	/*! The subsections of the report, altogether containing the different modes
	 *  that the algorithm is run in.
	 */
	subsection_t subsections[SECTION_SUBSECTION_COUNT];
} section_t;


/*!	\brief Executes the data-gathering phase for the specified section.
 *
 *	\return	NULL on success, exception_t* on failure.
 */
exception_t* section_execute(section_t* section, char* input_filepath);


/*!	\brief Iniitialize the specified section_t with the specified algorithm.
 *
 *	\return	NULL on success, exception_t* on failure.
 */
exception_t* section_init(section_t* section, int data_count, enum algorithm algorithm);


/*!	\brief Uninitialize the specified section_t.
 *
 *	\return	NULL on success, exception_t* on failure.
 */
exception_t* section_free(section_t* section);


/*!	\brief Write the appendix data of specified section_t to the specified file.
 *	
 *	\return	NULL on success, exception_t* on failure.
 */
exception_t* section_write_appendix(section_t* section, FILE* file);


/*!	\brief Write the results of the specified section_t* to the specified file.
 *
 *	\return	NULL on success, exception_t* on failure.
 */
exception_t* section_write_results_summary(section_t* section, FILE* file, off_t size);


/*!	\brief Write the gain results of the speicfied section_t* to the specified file.
 * 
 *	\return	NULL on success, exception_t* on failure.
 */
exception_t* section_write_results_gain(section_t* section, FILE* file, off_t size);


/*!	\brief Write the results table for either the encrypt or decrypt version from the specified section.
 *
 *	\return	NULL on success, exception-t* on failure.
 */
exception_t* section_write_results_summary_table(section_t* section, FILE* file, off_t size, enum encryption encryption);


#endif // section_H
