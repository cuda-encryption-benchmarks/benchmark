#include "section.h"


exception_t* section_init(section_t* section, enum algorithm algorithm, enum mode mode) {
	char* function_name = "section_init()";

	return exception_throw("Not implemented.", function_name);
}


exception_t* section_free(section_t* section) { 
	char* function_name = "section_free()";

	return exception_throw("Not implemented.", function_name);
}


exception_t* section_write(section_t* section, FILE* file) {
	char* function_name = "section_write()";

	return exception_throw("Not implemented.", function_name);
}

