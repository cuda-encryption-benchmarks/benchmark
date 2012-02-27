
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>

#include "block128.h"
#include "ccc/ccc.h"
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
	block128* input_blocks;
	block128* user_key;
	int block_count;
	int ifd;
	int ofd;

	// Validate the arguments.
	exception = arguments_validate(argc, argv);
	if ( exception != NULL ) {
		exception_catch(exception);
		exit(EXIT_FAILURE);
	}

	// Open input file.
	ifd = open(argv[2], O_RDONLY);
	if ( ifd == -1 ) {
		perror("Opening input file failed.\n");
		exit(EXIT_FAILURE);
	}

	// Open output file.
	ofd = open(argv[3], O_CREAT | O_TRUNC | O_WRONLY , 0700);
	if ( ofd == -1 ) {
		perror("Opening output file failed.\n");
		exit(EXIT_FAILURE);
	}

	// Read input file.
	exception = block128_read_file(ifd, &input_blocks, &block_count);
	if ( exception != NULL ) {
		exception_catch(exception);
		exit(EXIT_FAILURE);
	}

	// Encrypt input file.
	user_key = NULL;
	exception = serpent_encrypt_serial(user_key, input_blocks, block_count);
	if ( exception != NULL ) {
		exception_catch(exception);
		exit(EXIT_FAILURE);
	}

	// Write output file.
	exception = block128_write_file(ofd, input_blocks, block_count);
	if ( exception != NULL ) {
		exception_catch(exception);
		exit(EXIT_FAILURE);
	}

	// Close input file.
	ifd = close(ifd);
	if ( ifd == -1 ) {
		perror("Closing input file failed.\n");
		exit(EXIT_FAILURE);
	}

	// Close output file.
	ofd = close(ofd);
	if ( ofd == -1 ) {
		perror("Closing output file failed.\n");
		exit(EXIT_FAILURE);
	}

	// Return success.
	exit(EXIT_SUCCESS);

}
