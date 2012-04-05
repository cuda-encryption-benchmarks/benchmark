
// Macro to allow clock_getres() and related functions.
#define _POSIX_C_SOURCE 199309L

#include <errno.h>
#include <fcntl.h>
#include <time.h>
#include <unistd.h>

#include "aes.h"
#include "block128.h"
#include "ccc/ccc.h"
#include "file.h"
#include "mirror_bytes.h"
#include "serpent.h"
//#include "report/report.h" // DEBUG
#include "typedef.h"


void print_usage(char* message) {
	// Print usage message.
	fprintf(stdout, "\a\nERROR: %s\n\n", message);
	fprintf(stdout, "--- CUDA Benchmarks ---\n");
	fprintf(stdout, "USAGE: ./benchmarks <algorithm> <mode> <encryption> input\n");
	fprintf(stdout, "\talgorithm: The algorithm to encrypt/decrypt the file with.\n\t\t-Possible values: {aes,serpent,twofish}\n");
	fprintf(stdout, "\tmode: How to run the algorithm.\n\t\t-Possible values: {cuda,parallel,serial}\n");
	fprintf(stdout, "\tencryption: Whether to encrypt or decrypt the file.\n\t\t-Possible values: {decrypt,encrypt}\n");
	fprintf(stdout, "\tinput: The input file.\n");

	// Exit the program.
	exit(EXIT_FAILURE);
}


exception_t* arguments_parse(char* argv[], enum algorithm* algorithm, enum mode* mode, enum encryption* encryption) {
	exception_t* exception;
	char* function_name = "arguments_parse()";

	// Validate parameters.
	if ( argv == NULL ) {
		return exception_throw("argv was NULL.", function_name);
	}

	// Parse the algorithm.
	exception = arguments_parse_algorithm(argv[1], algorithm);
	if ( exception != NULL ) {
		return exception_append(exception, function_name);
	}

	// Parse the processing mode.
	exception = arguments_parse_mode(argv[2], mode);
	if ( exception != NULL ) {
		return exception_append(exception, function_name);
	}

	// Parse the encrypt/decrypt.
	exception = arguments_parse_encryption(argv[3], encryption);
	if ( exception != NULL ) {
		return exception_append(exception, function_name);
	}

	// Return success.
	return NULL;
}


exception_t* arguments_parse_algorithm(char* argument, enum algorithm* algorithm) {
	char* function_name = "arguments_parse_algorithm()";

	// Validate parameters.
	if ( argument == NULL ) {
		return exception_throw("argument was NULL.", function_name);
	}
	if ( algorithm == NULL ) {
		return exception_throw("algorithm was NULL.", function_name);
	}

	// Compare the strings.
	if ( strcmp(argument, "aes") == 0 ) {
		(*algorithm) = AES;
	}
	else if ( strcmp(argument, "serpent") == 0 ) {
		(*algorithm) = SERPENT;
	}
	else if ( strcmp(argument, "twofish") == 0 ) {
		(*algorithm) = TWOFISH;
	}
	else {
		print_usage("Invalid encryption algorithm.");
	}

	// Return success.
	return NULL;
}


exception_t* arguments_parse_encryption(char* argument, enum encryption* encryption) {
	char* function_name = "arguments_parse_encryption()";

	// Validate parameters.
	if ( argument == NULL ) {
		return exception_throw("argument was NULL.", function_name);
	}
	if ( encryption == NULL ) {
		return exception_throw("encryption was NULL.", function_name);
	}

	// Compare the strings.
	if ( strcmp(argument, "encrypt") == 0 ) {
		(*encryption) = ENCRYPT;
	}
	else if ( strcmp(argument, "decrypt") == 0 ) {
		(*encryption) = DECRYPT;
	}
	else {
		print_usage("Invalid encrypt/decrypt parameter.");
	}

	// Return success.
	return NULL;
}


exception_t* arguments_parse_mode(char* argument, enum mode* mode) {
	char* function_name = "arguments_parse_mode()";

	// Validate parameters.
	if ( argument == NULL ) {
		return exception_throw("argument was NULL.", function_name);
	}
	if ( mode == NULL ) {
		return exception_throw("mode was NULL.", function_name);
	}

	// Comparse the strings.
	if ( strcmp(argument, "cuda") == 0 ) {
		(*mode) = CUDA;
	}
	else if ( strcmp(argument, "parallel") == 0 ) {
		(*mode) = PARALLEL;
	}
	else if ( strcmp(argument, "serial") == 0 ) {
		(*mode) = SERIAL;
	}
	else {
		print_usage("Invalid mode.");
	}

	// Return success.
	return NULL;
}


exception_t* arguments_validate(int argc, char* argv[]) {
	//char* function_name = "arguments_validate()";

	// Validate argument count.
	if ( argc != 5 ) {
		print_usage("Invalid number of arguments.");
	}

	// Return success.
	return NULL;
}


int main( int argc, char* argv[] ) {
	exception_t* exception;
	block128_t* blocks;
	key256_t key;
	struct timespec clock_begin;
	struct timespec clock_end;
	long long block_count;
	int blocks_read;
	file_t file;
	enum algorithm algorithm;
	enum encryption encryption;
	enum mode mode;


	// DEBUG
	/*
	report_t report;
	exception = report_init(&report);
	if ( exception != NULL ) {
		exception_catch(exception);
		exit(EXIT_FAILURE);
	}

	exception = report_write(&report);
	if ( exception != NULL ) {
		exception_catch(exception);
		exit(EXIT_FAILURE);
	}

	exception = report_free(&report);
	if ( exception != NULL ) {
		exception_catch(exception);
		exit(EXIT_FAILURE);
	} */
	

	// Validate the arguments.
	exception = arguments_validate(argc, argv);
	if ( exception != NULL ) {
		exception_catch(exception);
		exit(EXIT_FAILURE);
	}

	// Parse the arguments.
	exception = arguments_parse(argv, &algorithm, &mode, &encryption);
	if ( exception != NULL ) {
		exception_catch(exception);
		exit(EXIT_FAILURE);
	}

	// Manually set the key.
	key.key0.x0 = 0xdeadbeef;
	key.key0.x1 = 0xdeadbeef;
	key.key0.x2 = 0xdeadbeef;
	key.key0.x3 = 0xdeadbeef;
	key.key1.x0 = 0xdeadbeef;
	key.key1.x1 = 0xdeadbeef;
	key.key1.x2 = 0xdeadbeef;
	key.key1.x3 = 0xdeadbeef;

	// Open input file.
	if ( encryption == ENCRYPT ) {
		exception = file_init(argv[4], UNENCRYPTED, &file);
	}
	else if ( encryption == DECRYPT ) {
		exception = file_init(argv[4], ENCRYPTED, &file);
	}
	else { 
		fprintf(stderr, "Unhandled encryption parameter.");
	}
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

	// Get clock resolution. clock_getres().
	fprintf(stdout, "Clock resolution: ");
	if ( clock_getres(CLOCK_REALTIME, &clock_begin) == -1 ) {
		fprintf(stdout, "Unknown -- ");
		fflush(stdout);
		perror(NULL);
	}
	else { 
		fprintf(stdout, "%li nanosecond(s)", clock_begin.tv_nsec );
	}
	fprintf(stdout, ".\n");

	// Get begin time.
	fprintf(stdout, "Begin time: ");
	if ( clock_gettime(CLOCK_REALTIME, &clock_begin) == -1 ) {
		fprintf(stdout, "Unknown -- ");
		fflush(stdout);
		perror(NULL);
	}
	else {
		fprintf(stdout, "%li seconds(s), %li nanosecond(s)", clock_begin.tv_sec, clock_begin.tv_nsec);
	}
	fprintf(stdout, ".\n");

	// Call the algorithm.
	switch(algorithm) {
	case AES:
		exception = aes(&key, blocks, block_count, mode, encryption);
		break;
	case SERPENT:
		exception = serpent(&key, blocks, block_count, mode, encryption);
		break;
	case TWOFISH:
		fprintf(stderr, "Not implemented.\n");
		break;
	default:
		fprintf(stderr, "ERROR: Unrecognized algorithm.");
		break;
	}
	if ( exception != NULL ) {
		exception_catch(exception);
		exit(EXIT_FAILURE);
	}

	// Get end time.
	fprintf(stdout, "End time: ");
	if ( clock_gettime(CLOCK_REALTIME, &clock_end) == -1 ) {
		fprintf(stdout, "Unknown -- ");
		fflush(stdout);
		perror(NULL);
	}
	else {
		fprintf(stdout, "%li second(s),%li nanosecond(s)", clock_end.tv_sec, clock_end.tv_nsec);
	}
	fprintf(stdout, ".\n");

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
	}

	// Return success.
	exit(EXIT_SUCCESS);

}
