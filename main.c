
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>

#include "block128.h"
#include "ccc/ccc.h"
#include "file.h"
#include "mirror_bytes.h"
#include "serpent.h"


/**	Enumeration representing which benchmark to run.
 */
enum benchmark_type {
	SERPENT_SERIAL,
	SERPENT_THREADED,
	SERPENT_CUDA,
	TWOFISH_SERIAL,
	TWOFISH_THREADED,
	TWOFISH_CUDA
};


/**	Print a usage message to stdout and exit the program.
 */
void print_usage(char* message) {
	// Print usage message.
	fprintf(stdout, "\a\nERROR: %s\n\n", message);
	fprintf(stdout, "--- CUDA Benchmarks ---\n");
	fprintf(stdout, "USAGE: ./benchmarks mode input output\n");
	fprintf(stdout, "\tmode: TODO\n");
	fprintf(stdout, "\tinput: The input file.\n");
	fprintf(stdout, "\toutput: The output file.\n\n");

	// Exit the program.
	exit(EXIT_FAILURE);
}


/**	Validate user-supplied arguments.
 *	@return	NULL on success, exception_t* on failure.
 */
exception_t* arguments_validate(int argc, char* argv[]) {
	//char* function_name = "arguments_validate()";

	// Validate argument count.
	if ( argc != 4 ) {
		print_usage("Invalid number of arguments.");
	}

	// Return success.
	return NULL;
}


int main( int argc, char* argv[] ) {
	//char* function_name = "main()";
	exception_t* exception;
	//block128* blocks;
	//block128* user_key;
	//long long block_count;
	//int blocks_read;
	//file_t file;

	uint32 in = 0xdeadbeef;
	uint32 out; // in, out, in, out, in, out... *guitar solo*

	// Print input
	fprintf(stdout, "Input: %X.\n", in);

	// Mirror 1
	exception = mirror_bytes32(in, &out);
	if ( exception != NULL ) {
		exception_catch(exception);
		exit(EXIT_FAILURE);
	}

	// Print output.
	fprintf(stdout, "Output: %X.\n", out);

	// Mirror 2
	exception = mirror_bytes32(out, &out);
	if ( exception != NULL ) {
		exception_catch(exception);
		exit(EXIT_FAILURE);
	}

	// Print input/output.
	fprintf(stdout, "Input again: %X.\n", out);


	/*
	// Validate the arguments.
	exception = arguments_validate(argc, argv);
	if ( exception != NULL ) {
		exception_catch(exception);
		exit(EXIT_FAILURE);
	}

	// Open input file.
	exception = file_init(argv[2], UNENCRYPTED, &file);
	if ( exception != NULL ) {
		exception_catch(exception);
		exit(EXIT_FAILURE);
	}

	// Get number of blocks in file.
	exception = file_get_block_count(&file, &block_count);
	if ( exception != NULL ) {
		exception_catch(exception);
		exit(EXIT_FAILURE);
	}

	// Read data from file.
	exception = file_read(&file, 0, block_count, &blocks, &blocks_read);
	if ( exception != NULL ) {
		exception_catch(exception);
		exit(EXIT_FAILURE);
	}

	// Modify data.
	for ( int i = 0; i < block_count; i++ ) {
		blocks[i].x0 =~ blocks[i].x0;
		blocks[i].x1 =~ blocks[i].x1;
		blocks[i].x2 =~ blocks[i].x2;
		blocks[i].x3 =~ blocks[i].x3;
	}

	// Write data to file.
	exception = file_write(&file, 0, block_count, blocks, &blocks_read);
	if ( exception != NULL ) {
		exception_catch(exception);
		exit(EXIT_FAILURE);
	}

	// Close input file.
	exception = file_free(&file);
	if ( exception != NULL ) {
		exception_catch(exception);
		exit(EXIT_FAILURE);
	} */

	// Return success.
	exit(EXIT_SUCCESS);

}
