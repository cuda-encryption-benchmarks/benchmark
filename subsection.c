
#include "subsection.h"


exception_t* subsection_execute(subsection_t* subsection, key256_t* key, char* input_filepath, enum algorithm algorithm) {
	char* function_name = "subsection_execute()";
	exception_t* exception;
	char mode_name[MODE_NAME_LENGTH];
	int i;

	// Validate parameters.
	if ( subsection == NULL ) {
		return exception_throw("subsection was NULL.", function_name);
	}
	if ( key == NULL ) {
		return exception_throw("key was NULL.", function_name);
	}
	if ( input_filepath == NULL ) {
		return exception_throw("input_filepath was NULL.", function_name);
	}

	// Get mode name.
	exception = mode_get_name(subsection->mode, mode_name);
	if ( exception != NULL ) {
		return exception_append(exception, function_name);
	}

	// Execute the algorithm.
	fprintf(stdout, "    Subsection \"%s\" execution starting.\n", mode_name);
	for ( i = 0; i < subsection->data_count; i++ ) {
		fprintf(stdout, "      Iteration #%i: ", i + 1);
		fflush(stdout);
		// Encrypt the file.
		exception = benchmark(key, input_filepath, algorithm, subsection->mode, ENCRYPT, &(subsection->data_encrypt[i]));
		if ( exception != NULL ) {
			return exception_append(exception, function_name);
		}

		// Decrypt the file.
		exception = benchmark(key, input_filepath, algorithm, subsection->mode, DECRYPT, &(subsection->data_decrypt[i]));
		if ( exception != NULL ) {
			return exception_append(exception, function_name);
		}

		// TODO: Checksum the file.

		fprintf(stdout, "\n");
	}

	// Return success.
	fprintf(stdout, "    Subsection \"%s\" execution completed successfully.\n", mode_name);
	return NULL;
}


exception_t* subsection_init(subsection_t* subsection, int data_count, enum mode mode) {
	char* function_name = "subsection_init()";
	int i;

	// Validate parameters.
	if ( subsection == NULL ) {
		return exception_throw("subsection was NULL.", function_name);
	}
	if ( data_count < 1 ) {
		return exception_throw("data_count must be > 0.", function_name);
	}

	// Set the mode.
	subsection->mode = mode;

	// Set the data count.
	subsection->data_count = data_count;

	// Allocate space for encryption data.
	subsection->data_encrypt = (benchmark_data_t*)malloc(sizeof(benchmark_data_t) * data_count);
	if ( subsection->data_encrypt == NULL ) {
		return exception_throw("Unable to allocate encryption data.", function_name);
	}

	// Allocate space for decryption data.
	subsection->data_decrypt = (benchmark_data_t*)malloc(sizeof(benchmark_data_t) * data_count);
	if ( subsection->data_decrypt == NULL ) {
		return exception_throw("Unable to allocate decryption data.", function_name);
	}

	// Initialize the data to have impossible values.
	for ( i = 0; i < data_count; i++ ) {
		subsection->data_encrypt[i].time_elapsed.tv_sec = -1;
		subsection->data_encrypt[i].time_elapsed.tv_nsec = -1;
		subsection->data_encrypt[i].buffer_size = -1;
		subsection->data_decrypt[i].time_elapsed.tv_sec = -1;
		subsection->data_decrypt[i].time_elapsed.tv_nsec = -1;
		subsection->data_decrypt[i].buffer_size = -1;
	}

	// Return success.
	return NULL;
}


exception_t* subsection_free(subsection_t* subsection) {
	char* function_name = "subsection_free()";
	int i;

	// Validate parameters.
	if ( subsection == NULL ) {
		return exception_throw("subsection was NULL.", function_name);
	}

	// Set mode to an impossible value.
	subsection->mode = -1;

	// Set the data to impossible values.
	for ( i = 0; i < subsection->data_count; i++ ) {
		subsection->data_encrypt[i].time_elapsed.tv_sec = -1;
		subsection->data_encrypt[i].time_elapsed.tv_nsec = -1;
		subsection->data_encrypt[i].buffer_size = -1;
		subsection->data_decrypt[i].time_elapsed.tv_sec = -1;
		subsection->data_decrypt[i].time_elapsed.tv_nsec = -1;
		subsection->data_decrypt[i].buffer_size = -1;
	}

	// Free the encryption data.
	free(subsection->data_encrypt);
	subsection->data_encrypt = NULL;

	// Free the decryption data.
	free(subsection->data_decrypt);
	subsection->data_decrypt = NULL;

	// Return success.
	return NULL;
}


exception_t* subsection_write(subsection_t* subsection, FILE* file, enum algorithm algorithm) {
	char* function_name = "subsection_write()";
	char mode_name[MODE_NAME_LENGTH];
	exception_t* exception;

	// Validate parameters.
	if ( subsection == NULL ) {
		return exception_throw("subsection was NULL.", function_name);
	}
	if ( file == NULL ) {
		return exception_throw("file was NULL.", function_name);
	}

	// Get mode name.
	exception = mode_get_name(subsection->mode, mode_name);
	if ( exception != NULL ) {
		return exception_append(exception, function_name);
	}
	fprintf(stdout, "    Subsection \"%s\" writing starting.\n", mode_name);

	// Write CSV output.
	fprintf(stdout, "      Writing CSV data: ");
	exception = subsection_write_csv(subsection, algorithm);
	if ( exception != NULL ) {
		fprintf(stdout, "FAILED: %s", exception->message);
	}
	fprintf(stdout, "\n");

	// Write LaTeX output.
	fprintf(stdout, "      Writing LaTeX data: ");
	exception = subsection_write_latex(subsection, file, algorithm);
	if ( exception != NULL ) {
		fprintf(stdout, "FAILED: %s", exception->message);
	}
	fprintf(stdout, "\n");

	// Return success.
	fprintf(stdout, "    Subsection \"%s\" written.\n", mode_name);
	return NULL;
}


exception_t* subsection_write_csv(subsection_t* subsection, enum algorithm algorithm) {
	char* function_name = "subsection_write_csv()";
	exception_t* exception;

	// Validate parameters.
	if ( subsection == NULL ) {
		return exception_throw("subsection was NULL.", function_name);
	}

	// Write encryption data.
	fprintf(stdout, "Encryption. ");
	exception = subsection_write_csv_file(subsection->data_encrypt, subsection->data_count, algorithm, subsection->mode, ENCRYPT);
	if ( exception != NULL ) {
		return exception_append(exception, function_name);
	}

	// Write decryption data.
	fprintf(stdout, "Decryption. ");
	exception = subsection_write_csv_file(subsection->data_decrypt, subsection->data_count, algorithm, subsection->mode, DECRYPT);
	if ( exception != NULL ) {
		return exception_append(exception, function_name);
	}

	// Return success.
	return NULL;
}


exception_t* subsection_write_csv_file(benchmark_data_t* data, int data_count, enum algorithm algorithm, enum mode mode, enum encryption encryption) {
	char* function_name = "subsection_write_csv_file()";
	exception_t* exception;
	FILE* csv_file;
	char algorithm_name[ALGORITHM_NAME_LENGTH];
	char mode_name[MODE_NAME_LENGTH];
	char encryption_name[ENCRYPTION_NAME_LENGTH];
	char csv_filename[ALGORITHM_NAME_LENGTH + MODE_NAME_LENGTH + ENCRYPTION_NAME_LENGTH + 4];
	int i;

	// Validate parameters.
	if ( data == NULL ) {
		return exception_throw("data was NULL.", function_name);
	}
	if ( data_count < 1 ) {
		return exception_throw("data_count less than 1.", function_name);
	}

	// Get the algorithm name.
	exception = algorithm_get_name(algorithm, algorithm_name);
	if ( exception != NULL ) {
		return exception_append(exception, function_name);
	}

	// Get the mode name.
	exception = mode_get_name(mode, mode_name);
	if ( exception != NULL ) {
		return exception_append(exception, function_name);
	}

	// Get the encryption name.
	exception = encryption_get_name(encryption, encryption_name);
	if ( exception != NULL ) {
		return exception_append(exception, function_name);
	}

	// Build the filename.
	strcpy(csv_filename, algorithm_name);
	strcat(csv_filename, "_");
	strcat(csv_filename, mode_name);
	strcat(csv_filename, "_");
	strcat(csv_filename, encryption_name);
	strcat(csv_filename, ".csv");

	// Convert filename to lower-case.
	for ( i = 0; i < strlen(csv_filename); i++ ) {
		csv_filename[i] = tolower(csv_filename[i]);
	}

	// Open the file
	csv_file = fopen(csv_filename, "w");
	if ( csv_file == NULL ) {
		return exception_throw("Unable to open CSV file.", function_name);
	}

	// Write CSV headers.
	fprintf(csv_file, "Iteration #, Seconds, Nanoseconds");
	if ( mode == CUDA ) {
		fprintf(csv_file, ", Global Memory Used");
	}
	fprintf(csv_file, "\n");

	// Write data.
	for ( i = 0; i < data_count; i++ ) {
		fprintf(csv_file, "%i,%li,%li", i + 1, data[i].time_elapsed.tv_sec, data[i].time_elapsed.tv_nsec);
		if ( mode == CUDA ) {
			fprintf(csv_file, ",%" PRIuPTR, data[i].buffer_size);
		}
		fprintf(csv_file, "\n");
	}

	// Close the file.
	if ( fclose(csv_file) == EOF ) {
		return exception_throw("Unable to close CSV file.", function_name);
	}

	// Return success.
	return NULL;
}


exception_t* subsection_write_latex(subsection_t* subsection, FILE* file, enum algorithm algorithm) {
	char* function_name = "subsection_write_latex()";
	exception_t* exception;
	char mode_name[MODE_NAME_LENGTH];

	// Validate parameters.
	if ( subsection == NULL ) {
		return exception_throw("subsection was NULL.", function_name);
	}
	if ( file == NULL ) {
		return exception_throw("file was NULL.", function_name);
	}

	// Get the mode of the subsection.
	exception = mode_get_name(subsection->mode, mode_name);
	if ( exception != NULL ) {
		return exception_append(exception, function_name);
	}

	// Write subsection head.
	fprintf(file, "\\subsubsection{%s}\n", mode_name);

	// Write encryption data.
	fprintf(stdout, "Encryption. ");
	exception = subsection_write_latex_table(file, subsection->data_encrypt, subsection->data_count, algorithm, subsection->mode, ENCRYPT);
	if ( exception != NULL ) {
		return exception_append(exception, function_name);
	}

	// Write decryption data.
	fprintf(stdout, "Decryption. ");
	exception = subsection_write_latex_table(file, subsection->data_decrypt, subsection->data_count, algorithm, subsection->mode, DECRYPT);
	if (exception != NULL ) {
		return exception_append(exception, function_name);
	}

	// Return success.
	return NULL;
}


exception_t* subsection_write_latex_table(FILE* file, benchmark_data_t* data, int data_count, enum algorithm algorithm, enum mode mode, enum encryption encryption) {
	char* function_name = "subsection_write_latex_table()";
	exception_t* exception;
	char algorithm_name[ALGORITHM_NAME_LENGTH];
	char mode_name[MODE_NAME_LENGTH];
	char encryption_name[ENCRYPTION_NAME_LENGTH];
	int i;

	// Validate parameters.
	if ( file == NULL ) {
		return exception_throw("file was NULL", function_name);
	}
	if ( data == NULL ) {
		return exception_throw("data was NULL", function_name);
	}

	// Get algorithm name.
	exception = algorithm_get_name(algorithm, algorithm_name);
	if ( exception != NULL ) {
		return exception_append(exception, function_name);
	}

	// Get mode name.
	exception = mode_get_name(mode, mode_name);
	if ( exception != NULL ) {
		return exception_append(exception, function_name);
	}

	// Get encryption name,
	exception = encryption_get_name(encryption, encryption_name);
	if ( exception != NULL ) {
		return exception_append(exception, function_name);
	}

	// Print the table head.
	fprintf(file, "\\begin{figure}[H]\n \
		\\caption{%s %s %s data}\n\\centering\n", algorithm_name, mode_name, encryption_name);

	// Print tabular head.
	fprintf(file, "\\begin{tabular}[c]{r|r|r");
	if ( mode == CUDA ) { // CUDA has one more entry.
		fprintf(file, "|r");
	}
	fprintf(file, "}\n");

	// Print row headers.
	fprintf(file, "Iteration \\# & Elapsed Seconds & Elapsed Nanoseconds ");
	if ( mode == CUDA ) {
		fprintf(file, "& Global Memory Used ");
	}
	fprintf(file, "\\\\\n\\hline\n");

	// Print the data.
	for ( i = 0; i < data_count; i++ ) {
		fprintf(file, "%i & %li & %li ", i + 1, data[i].time_elapsed.tv_sec, data[i].time_elapsed.tv_nsec);
		if ( mode == CUDA ) {
			fprintf(file, "& %" PRIuPTR, data[i].buffer_size);
		}
		fprintf(file, "\\\\\n");
	}

	// Print tabular and table tails.
	fprintf(file, "\\hline\n \
			\\end{tabular}\n\\end{figure}\n");

	// Return success.
	return NULL;
}

