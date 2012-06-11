#include "report.h"


exception_t* report_execute(report_t* report) {
	char* function_name = "report_execute()";
	exception_t* exception;
	int i;

	// Validate parameters.
	if ( report == NULL ) {
		return exception_throw("report was NULL.", function_name);
	}

	// Execute each subsection.
	fprintf(stdout, "Report execution starting.\n");
	for ( i = 0; i < REPORT_SECTION_COUNT; i++ ) {
		exception = section_execute(&(report->sections[i]), report->input_filepath);
		if ( exception != NULL ) {
			return exception_append(exception, function_name);
		}
	}

	// Return success.
	fprintf(stdout, "Report execution completed successfully!\n");
	return NULL;
}


exception_t* report_init(report_t* report, char* input_filepath, int data_count) {
	char* function_name = "report_init()";
	exception_t* exception;
	char filepath[REPORT_BASEPATH_LENGTH_MAX + REPORT_FILENAME_LENGTH_MAX + 3];
	int length;

	// Validate parameters.
	if ( report == NULL ) {
		return exception_throw("report was NULL.", function_name);
	}
	if ( input_filepath == NULL ) {
		return exception_throw("input_filepath was NULL.", function_name);
	}
	if ( data_count < 1 ) {
		return exception_throw("data_count must be > 0", function_name);
	}
	fprintf(stdout, "Initializing the report. ");

	// Set the report filename.
	strcpy(report->filename, "report");

	// Copy and set the input filename.
	length = strlen(input_filepath);
	if ( length == 0 ) {
		return exception_throw("Empty input filepath.", function_name);
	}
	report->input_filepath = (char*)malloc(sizeof(char) * (length + 1)); // +1 for the null character.
	if ( report->input_filepath == NULL ) {
		return exception_throw("Unable to malloc input filepath.", function_name);
	}
	strcpy(report->input_filepath, input_filepath);

	// Create the directory structure.
	report->basepath[0] = '\0';
	exception = report_init_create_directories(report);
	if ( exception != NULL ) {
		return exception_append(exception, function_name);
	}

	// Get the file stats.
	if ( stat64(input_filepath, &(report->stats)) == -1 ) { // File exists.
		return exception_throw("Unable to get file stats.", function_name);
	}

	// Open the report file.
	filepath[0] = '\0';
	strcat(filepath, report->basepath);
	strcat(filepath, report->filename);
	strcat(filepath, ".tex");
	report->file = fopen(filepath, "w");
	if ( report->file == NULL ) {
		return exception_throw("Unable to create report file.", function_name);
	}

	// Initialize each section of the report.
	// Serpent section.
	exception = section_init(&(report->sections[REPORT_SECTION_SERPENT]), data_count, SERPENT);
	if ( exception != NULL ) {
		return exception_append(exception, function_name);
	}
	// Twofish section.
	exception = section_init(&(report->sections[REPORT_SECTION_TWOFISH]), data_count, TWOFISH);
	if ( exception != NULL ) {
		return exception_append(exception, function_name);
	}

	// Return success.
	fprintf(stdout, "\n");
	return NULL;
}


exception_t* report_init_create_directories(report_t* report) {
	char* function_name = "report_init_create_directories()";
	struct stat stats;
	time_t time_raw;
	struct tm time_current;
	char time_buffer[50];

	// Validate parameters.
	if ( report == NULL ) {
		return exception_throw("report was NULL.", function_name);
	}

	// Make/ensure creation of results directory.
	strcpy(report->basepath, "runs");
	if ( stat(report->basepath, &stats) == 0 ) { // File exists.
		// Check if the file is a directory.
		if ( !(S_ISDIR(stats.st_mode)) ) {
			return exception_throw("The runs file exists and is not a directory.", function_name);
		}
	}
	else { // Stat failed.
		if ( errno == ENOENT ) { // File does not exist.
			// Create the directory.
			if ( mkdir(report->basepath, 0700) == -1 ) {
				perror(NULL);
				return exception_throw("Unable to create results directory.", function_name);
			}
		}
		else { // Any other error.
			// Return failure.
			perror(NULL);
			return exception_throw("Unable to check for results directory.", function_name);
		}
	}

	// Get the current time.
	if ( time(&time_raw) == -1 ) {
		return exception_throw("Unable to get current time.", function_name);
	}

	// Get the struct tm representation of the current time.
	if ( localtime_r(&time_raw, &time_current) == NULL ) {
		return exception_throw("Unable to get struct tm.", function_name);
	}

	// Get the string representation of the current time.
	if ( strftime(time_buffer, 50, "%Y_%m_%d_%H:%M:%S", &time_current) == 0 ) {
		return exception_throw("Unable to get string representation of current time.", function_name);
	}

	// Copy the string representation of the current time onto the report filepath.
	strcat(report->basepath, "/");
	strcat(report->basepath, time_buffer);
	strcat(report->basepath, "/");

	// Create the report directory.
	if ( mkdir(report->basepath, 0700) == -1 ) {
		perror(NULL);
		return exception_throw("Unable to create report directory.", function_name);
	}

	// Return success.
	return NULL;
}


exception_t* report_free(report_t* report) {
	char* function_name = "report_free()";
	exception_t* exception;
	int i;

	// Validate parameters.
	if ( report == NULL ) {
		return exception_throw("report was NULL.", function_name);
	}
	fprintf(stdout, "Wrapping up the report. ");

	// Invalidate the basepath.
	report->basepath[0] = '\0';

	// Invalidate the filename.
	report->filename[0] = '\0';

	// Free the input filepath.
	free(report->input_filepath);
	report->input_filepath = NULL;

	// Free each section of the report.
	for ( i = 0; i < REPORT_SECTION_COUNT; i++ ) {
		exception = section_free(&(report->sections[i]));
		if ( exception != NULL ) {
			return exception_append(exception, function_name);
		}
	}

	// Close the report file.
	if ( fclose(report->file) == EOF ) {
		return exception_throw("Unable to close report file.", function_name);
	}	

	// Return success.
	fprintf(stdout, "\n");
	return NULL;
}


exception_t* report_write(report_t* report) { 
	char* function_name = "report_write()";
	exception_t* exception;

	// Validate parameters.
	if ( report == NULL ) {
		return exception_throw("report was NULL.", function_name);
	}
	fprintf(stdout, "Report writing starting.\n");

	// Move into the report directory.
	fprintf(stdout, "Moving into report directory. ");
	if ( chdir(report->basepath) == -1 ) {
		fprintf(stdout, "ERROR: ");
		fflush(stdout);
		perror(NULL);
		return exception_throw("Unable to chdir into report directory.", function_name);
	}
	fprintf(stdout, "\n");

	// Write document head.
	fprintf(report->file, "\\documentclass{article}\n\n");
	fprintf(report->file, "\\usepackage{amsmath}\n\n");
	fprintf(report->file, "\\usepackage{float}\n\n");
	fprintf(report->file, "\\usepackage[left=2cm,right=2cm,top=2cm,bottom=3cm,nohead]{geometry}\n");
	fprintf(report->file, "\\title{CUDA Benchmarking Report}\n\\author{Automatically Generated}\n\n");
	fprintf(report->file, "\\begin{document}\n\n");
	fprintf(report->file, "\\maketitle\n\n");

	// Write introduction section. TODO
	fprintf(report->file, "\\section{Introduction}\n");
	fprintf(report->file, "This is an automatically-generated report for the CUDA benchmarking suite.\n\n");

	// Write methodologies.
	exception = report_write_methodologies(report);
	if ( exception != NULL ) {
		fprintf(stdout, "ERROR writing methodologies: %s\n", exception->message);
	}

	// Write results.
	exception = report_write_results(report);
	if ( exception != NULL ) {
		return exception_append(exception, function_name);
		fprintf(stdout, "ERROR writing results: %s\n", exception->message);
	}

	// Write system information.
	fprintf(stdout, "Writing system information. ");
	fflush(stdout);
	exception = report_write_system_information(report);
	if ( exception != NULL ) {
		fprintf(stdout, "ERROR writing system information: %s", exception->message);
	}
	fprintf(stdout, "\n");

	// Write the appendix.
	exception = report_write_appendix(report);
	if ( exception != NULL ) {
		fprintf(stdout, "ERROR writing results: %s\n", exception->message);
	}
	
	// Write document tail.
	fprintf(report->file, "\\end{document}\n");

	// Flush the file buffer.
	fprintf(stdout, "Flushing LaTeX buffer. ");
	if ( fflush(report->file) == EOF ) {
		fprintf(stdout, "ERROR.");
	}
	fprintf(stdout, "\n");

	// Compile the report into LaTeX.
	fprintf(stdout, "Compiling into LaTeX. ");
	fflush(stdout);
	exception = report_write_compile_latex(report);
	if ( exception != NULL ) {
		fprintf(stdout, "ERROR: %s", exception->message);
	}
	fprintf(stdout, "\n");

	// Move back into the original directory.
	fprintf(stdout, "Moving back into original directory. ");
	if ( chdir("../../") == -1 ) {
		fprintf(stdout, "ERROR: ");
		fflush(stdout);
		perror(NULL);
		return exception_throw("Unable to chdir into original directory.", function_name);
	}
	fprintf(stdout, "\n");

	// Return success.
	fprintf(stdout, "Report writing completed.\n");
	return NULL;
}


exception_t* report_write_appendix(report_t* report) {
	char* function_name = "report_write_appendix()";
	exception_t* exception;
	int i;

	// Validate parameters.
	if ( report == NULL ) {
		return exception_throw("report was NULL.", function_name);
	}

	// Write results section header.
	fprintf(report->file, "\\section{Appendix}\n");

	// Write each section of the report.
	for ( i = 0; i < REPORT_SECTION_COUNT; i++ ) {
		exception = section_write_appendix(&(report->sections[i]), report->file);
		if ( exception != NULL ) {
			return exception_append(exception, function_name);
		}
	}

	// Return success.
	return NULL;
}


exception_t* report_write_compile_latex(report_t* report) {
	char* function_name = "report_write_compile_latex()";
	char filename_dvi[REPORT_FILENAME_LENGTH_MAX + 4];
	char filename_ps[REPORT_FILENAME_LENGTH_MAX + 3];
	char filename_tex[REPORT_FILENAME_LENGTH_MAX + 4];
	int child_id;
	int child_status;
	int exit_status;
	int fd;
	int i;

	// Build the full filename.
	strcpy(filename_tex, report->filename);
	strcat(filename_tex, ".tex");

	// Compile the document in latex (twice).
	for ( i = 0; i < 2; i++ ) {
		// Fork the child.
		child_id = fork();
		if ( child_id == -1 ) {
			return exception_throw("Unable to fork latex child.", function_name);
		}

		// Exec into latex.
		if ( child_id == 0 ) {
			// Open a gateway to the void in-between terminals.
			fd = open("/dev/null", O_WRONLY);
			if ( fd == -1 ) {
				perror("Failed to open /dev/null; incoming nastiness...");
			}
			else {
				// Close stdout and stderr.
				if ( close(STDOUT_FILENO) == -1 ) {
					perror("Unable to close LaTeX stdout.");
				}
				if ( close(STDERR_FILENO) == -1 ) {
					perror("Unable to close LaTeX stderr.");
				}

				// Banish stdout and stderr from this virtual world.
				if ( dup2(fd, STDOUT_FILENO) == -1 ) {
					perror("Unable to dup2 stdout.");
				}
				if ( dup2(fd, STDERR_FILENO) == -1 ) {
					perror("Unable to dup2 stderr.");
				}
			}

			// Execute latex.
			execlp("latex", "latex", "-interaction=nonstopmode", filename_tex, (char*)'\0');
			return exception_throw("Unable to exec into latex.", function_name);
		}

		// Wait for the child.
		if ( waitpid(child_id, &child_status, 0) == -1 ) {
			return exception_throw("Unable to wait for latex child.", function_name);
		}

		// Check the child exit status.
		if ( WIFEXITED(child_status) ) {
			exit_status = WEXITSTATUS(child_status);
			if ( exit_status == EXIT_FAILURE ) {
				return exception_throw("latex child returned EXIT_FAILURE.", function_name);
			}
		}
		else { 
			return exception_throw("latex child did not exit properly.", function_name);
		}
	}

	// Build the DVI filename.
	strcpy(filename_dvi, report->filename);
	strcat(filename_dvi, ".dvi");

	// Build the PS filename.
	strcpy(filename_ps, report->filename);
	strcat(filename_ps, ".ps");

	// Convert the DVI file into a PS file.
	child_id = fork();
	if ( child_id == -1 ) {
		return exception_throw("Unable to fork dvips child.", function_name);
	}
	if ( child_id == 0 ) {
		// Open a gateway to the void in-between terminals.
		fd = open("/dev/null", O_WRONLY);
		if ( fd == -1 ) {
			perror("Failed to open /dev/null; incoming nastiness...");
		}
		else {
			// Close stdout and stderr.
			if ( close(STDOUT_FILENO) == -1 ) {
				perror("Unable to close LaTeX stdout.");
			}
			if ( close(STDERR_FILENO) == -1 ) {
				perror("Unable to close LaTeX stderr.");
			}

			// Banish stdout and stderr from this virtual world.
			if ( dup2(fd, STDOUT_FILENO) == -1 ) {
				perror("Unable to dup2 stdout.");
			}
			if ( dup2(fd, STDERR_FILENO) == -1 ) {
				perror("Unable to dup2 stderr.");
			}
		}

		// Execute dvips.
		execlp("dvips", "dvips", "-R", "-Poutline", "-t", "letter", filename_dvi, "-o", filename_ps, (char*)'\0');
		return exception_throw("Unable to exec into dvips.", function_name);
	}

	// Wait for the dvips child.
	if ( waitpid(child_id, &child_status, 0) == -1 ) {
		return exception_throw("Unable to wait for dvips child.", function_name);
	}

	// Check the child exit status.
	if ( WIFEXITED(child_status) ) {
		exit_status = WEXITSTATUS(child_status);
		if ( exit_status == EXIT_FAILURE ) {
			return exception_throw("dvips child returned EXIT_FAILURE.", function_name);
		}
	}
	else { 
		return exception_throw("dvips child did not exit properly.", function_name);
	}

	// Convert the PS file into a PDF file.
	child_id = fork();
	if ( child_id == -1 ) {
		return exception_throw("Unable to fork ps2pdf child.", function_name);
	}
	if ( child_id == 0 ) {
		// Open a gateway to the void in-between terminals.
		fd = open("/dev/null", O_WRONLY);
		if ( fd == -1 ) {
			perror("Failed to open /dev/null; incoming nastiness...");
		}
		else {
			// Close stdout and stderr.
			if ( close(STDOUT_FILENO) == -1 ) {
				perror("Unable to close LaTeX stdout.");
			}
			if ( close(STDERR_FILENO) == -1 ) {
				perror("Unable to close LaTeX stderr.");
			}

			// Banish stdout and stderr from this virtual world.
			if ( dup2(fd, STDOUT_FILENO) == -1 ) {
				perror("Unable to dup2 stdout.");
			}
			if ( dup2(fd, STDERR_FILENO) == -1 ) {
				perror("Unable to dup2 stderr.");
			}
		}	

		// Execute ps2pdf.
		execlp("ps2pdf", "ps2pdf", filename_ps, (char*)'\0');
		return exception_throw("Unable to exec into ps2pdf.", function_name);
	}

	// Wait for the ps2pdf child.
	if ( waitpid(child_id, &child_status, 0) == -1 ) {
		return exception_throw("Unable to wait for ps2pdf child.", function_name);
	}

	// Check the child exit status.
	if ( WIFEXITED(child_status) ) {
		exit_status = WEXITSTATUS(child_status);
		if ( exit_status == EXIT_FAILURE ) {
			return exception_throw("ps2pdf child returned EXIT_FAILURE.", function_name);
		}
	}
	else { 
		return exception_throw("ps2pdf child did not exit properly.", function_name);
	}

	// Return success.
	return NULL;
}


exception_t* report_write_methodologies(report_t* report) {
	char* function_name = "report_write_methodologies()";

	// Validate parameters.
	if ( report == NULL ) {
		return exception_throw("report was NULL.", function_name);
	}

	// Write section header.
	fprintf(report->file, "\\section{Methodologies}\n");

	// Write benchmark timing.
	fprintf(report->file, "\\subsection{Benchmark Timing}\n" \
		"Algorithm benchmarking is done by reading the entire file " \
		"into memory, starting the benchmark time, running the algorithm on the file, " \
		"stopping the benchmark time, and writing the file back to disk. This " \
		"eliminates disk I/O as a factor of uncertainty in the benchmarking time.\n");

	// Write clock resolution using clock_getres().
	fprintf(report->file, "\\subsection{Clock Resolution}\n" \
		"Benchmarking times are measured using the \\verb'clock_getres()' function. " \
		"Note that the accuracy of this method may be system-dependent.\n");

	// Write the harmonic mean.
	fprintf(report->file, "\\subsection{Harmonic Mean}\n" \
		"The harmonic mean $H$ is calculated by the following standard procedure:\n" \
		"\\begin{eqnarray*}\n" \
		"H & = & \\frac{n}{\\frac{1}{x_1} + \\frac{1}{x_2} + \\cdots + \\frac{1}{x_n}}\n" \
		"\\end{eqnarray*}\n");

	// Write parallelization (Open MP).
	fprintf(report->file,"\\subsection{Parallelization}\n" \
		"The method used for parallelization of the algorithms is OpenMP. " \
		"The algorithm makes use of every logical processor on the machine. ");

	// Write standard deviation.
	fprintf(report->file, "\\subsection{Sample Standard Deviation}\n" \
		"The sample standard deviation is calculated using the adjusted version " \
		"for sample standard deviation, defined as follows:\n" \
		"\\begin{eqnarray*}\n" \
		"s & = & \\sqrt{\\frac{1}{N - 1}\\sum_{i = 1}^{N}(x_i - \\overline{x})^2}\n" \
		"\\end{eqnarray*}\n" \
		"where $N$ is the population count, $x_i$ is the ith sample, and $\\overline{x}$ " \
		"is the population mean.\n\n" \
		"Internally, these values are calculated by converting the benchmark second and nanosecond " \
		"integer values into double-precision floating-point values. This results in a minor " \
		"loss of precision, which should be negligible in all but extreme cases " \
		"(such as having a value which is orders of magnitude larger than the others, " \
		" or millions of iterations).\n");

	// Write the sample mean
	fprintf(report->file, "\\subsection{Sample Mean}\n" \
		"The sample mean is calculated by the standard formula:\n" \
		"\\begin{eqnarray*}\n" \
		"\\overline{x} & = & \\frac{1}{N} \\sum_{i = 1}^{N} x_{i}\n" \
		"\\end{eqnarray*}\n");

	// Return success.
	return NULL;
}


exception_t* report_write_results(report_t* report) {
	char* function_name = "report_write_results()";
	exception_t* exception;
	int i;

	// Validate parameters.
	#ifdef DEBUG_REPORT
	if ( report == NULL ) {
		return exception_throw("report was NULL.", function_name);
	}
	#endif
	
	// Write results header.
	fprintf(report->file, "\\section{Results}\n");

	// Write file size.
	fprintf(report->file, "The following results were generated from " \
		"a file of size:\n\\begin{verbatim}\n" \
		"%" PRIuMAX " bytes\n" \
		"\\end{verbatim}\n", report->stats.st_size);

	// Write total gains.
	exception = report_write_results_gain(report);
	if ( exception != NULL ) {
		return exception_append(exception, function_name);
	}

	// Write results from each section.
	for ( i = 0; i < REPORT_SECTION_COUNT; i++ ) {
		exception = section_write_results_summary(&(report->sections[i]), report->file, report->stats.st_size);
		if ( exception != NULL ) {
			return exception_append(exception, function_name);
		}
	}

	// Return success.
	return NULL;
}


exception_t* report_write_results_gain(report_t* report) {
	char* function_name = "report_write_results_gain()";
	exception_t* exception;
	int i;

	// Validate parameters.
	#ifdef DEBUG_REPORT
	if ( report == NULL ) {
		return exception_throw("report was NULL.", function_name);
	}
	#endif

	// Write the gain header.
	fprintf(report->file, "\\subsection{Gains}\n");

	// Write gain explaination.
	fprintf(report->file,"Gain ``speeds'' are measured by number of 128-bit blocks-per-second. The gains are relative to the serial mode.\n");

	// Write the table head.
	fprintf(report->file, "\\begin{figure}[H]\n" \
		"\\caption{Overall Gains}\n\\centering\n");

	// Write the tabular head.
	fprintf(report->file, "\\begin{tabular}[c]{|c|c|c|c|c|c|}\n" \
		"Algorithm & Serial Speed & Parallel Speed & CUDA Speed & Parallel Speed & CUDA Gains\\\\\n\\hline\n");

	// Write tabular rows from each section.
	for ( i = 0; i < REPORT_SECTION_COUNT; i++ ) {
		exception = section_write_results_gain(&(report->sections[i]), report->file, report->stats.st_size);
		if ( exception != NULL ) {
			return exception_append(exception, function_name);
		}
	}

	// Write the tabular and table tails.
	fprintf(report->file, "\\hline\\end{tabular}\n\\end{figure}\n");

	// Return success.
	return NULL;
}


exception_t* report_write_system_information(report_t* report) {
	char* function_name = "report_write_system_information()";
	exception_t* exception;
	struct timespec timespec;

	// Validate parameters.
	if ( report == NULL ) {
		return exception_throw("report was NULL.", function_name);
	}

	// Write system information header.
	fprintf(report->file, "\\section{System Information}\n");

	// Write clock resolution.
	fprintf(report->file, "\\subsection{Clock Resolution}\n" \
		"The resolution of this clock is:\n\\begin{verbatim}\n");
	if ( clock_getres(CLOCK_REALTIME, &timespec) == -1 ) {
		fprintf(report->file, "Unknown");
		fprintf(stdout, "ERROR getting clock resolution: ");
		fflush(stdout);
		perror(NULL);
	}
	else {
		fprintf(report->file, "%li nanosecond(s)", timespec.tv_nsec );
	}
	fprintf(report->file,"\n\\end{verbatim}\n\n");

	// Write CUDA devices
	exception = report_write_system_information_cuda_devices(report);
	if ( exception != NULL ) {
		return exception_append(exception, function_name);
	}

	// Write processor count.
	fprintf(report->file, "\\subsection{Processor Count}\n" \
		"The number of logical processors on this machine is:\n\\begin{verbatim}\n");
	#if defined(_OPENMP)
		fprintf(report->file, "%i", omp_get_num_procs());
	#else
		fprintf(report->file, "Unknown");
	#endif
	fprintf(report->file,"\n\\end{verbatim}\n\n");

	// Write all hardware information.
	exception = report_write_system_information_lshw(report);
	if ( exception != NULL ) {
		return exception_append(exception, function_name);
	}

	// Return success.
	return NULL;
}


exception_t* report_write_system_information_cuda_devices(report_t* report) {
	char* function_name = "report_write_system_information_cuda_devices()";
	int device_count;
	int i;
	
	// Validate parameters.
	if ( report == NULL ) {
		return exception_throw("report was NULL.", function_name);
	}

	// Write section header.
	fprintf(report->file, "\\subsection{CUDA Devices}\n" \
		"Note that only the first device found is used for the algorithms.\n");

	// Get number of CUDA devices.
	if ( cuda_device_count(&device_count) == -1 ) {
		return exception_throw("Unable to get device count.", function_name);
	}

	// Write each device. 
	if ( device_count == 0 ) {
		fprintf(report->file, "No CUDA-capable devices found.\n");
	}
	else {
		for ( i = 0; i < device_count; i++ ) {
			if ( cuda_device_properties_report_write(report->file, i) == -1 ) {
				return exception_throw("Unable to write device properties.", function_name);
			}
		}
	}

	// Return success.
	return NULL;
}


exception_t* report_write_system_information_lshw(report_t* report) {
	char* function_name = "report_write_system_information_lshw()";
	char* filename = "hardware.txt";
	int child_pid;
	int child_status;
	int fd;

	// Validate parameters.
	if ( report == NULL ) {
		return exception_throw("report was NULL.", function_name);
	}

	// Write lshw header.
	fprintf(report->file, "\\subsection{All Hardware}\n" \
		"More hardware information is generated by running the \\verb'lshw' command and is stored in the ``hardware.txt'' file. " \
		"If the \\verb'lshw' command is not installed then, obviously, that hardware information will not be generated. " \
		"Note that running the \\verb'lshw' command as root will generate the most information, " \
		"and should be preferred to running \\verb'lshw' as a regular user; this would mean running " \
		"the entire program as root, if possible.\n\n");

	// Open a file for system hardware information.
	fd = open(filename, O_CREAT | O_WRONLY | O_TRUNC, 0600);
	if ( fd == -1 ) {
		perror(NULL);
		return exception_throw("Unable to open file for hardware data.", function_name);
	}

	// Fork a child to run lshw.
	child_pid = fork();
	if ( child_pid == -1 ) {
		return exception_throw("Unable to fork child.", function_name);
	}
	else if ( child_pid == 0 ) {
		// Close stdout.
		if ( close(STDOUT_FILENO) == -1 ) {
			return exception_throw("Unable to close stdout.", function_name);
		}

		// Close stderr.
		if ( close(STDERR_FILENO) == -1 ) {
			return exception_throw("Unable to close stderr.", function_name);
		}

		// Duplicate stdout to the opened file.
		if ( dup2(fd, STDOUT_FILENO) == -1 ) {
			return exception_throw("Unable to dup2 stdout to file.", function_name);
		}

		// Duplicate stderr to the opened file.
		if ( dup2(fd, STDERR_FILENO) == -1 ) {
			return exception_throw("Unable to dup2 stderr to file.", function_name);
		}

		// Close the extra fd.
		if ( close(fd) == -1 ) {
			return exception_throw("Unable to close fd.", function_name);
		}

		// Exec into lshw
		execlp("lshw", "lshw", (char*)'\0');
		return exception_throw("Unable to exec into lshw.", function_name);
	}
	
	// Close the opened file.
	if ( close(fd) == -1 ) {
		return exception_throw("Unable to close hardware data file.", function_name);
	}

	// Wait for the child.
	if ( waitpid(child_pid, &child_status, 0) == -1 ) {
		return exception_throw("Unable to wait for child.", function_name);
	}

	// Check the child exit status.
	if ( WIFEXITED(child_status) ) {
		int exit_status;
		exit_status = WEXITSTATUS(child_status);
		if ( exit_status == EXIT_FAILURE ) {
			return exception_throw("Child returned EXIT_FAILURE.", function_name);
		}
	}
	else { 
		return exception_throw("Child did not exit properly.", function_name);
	}

	// PROBLEM: Wiritng in a LaTeX verbatim environment causes the output to go
	// off the end of the file, and special characters mean that writing into a normal
	// environment in not easy. Not sure of a better way to implement this in time, so
	// the separate hardware.txt file will have to do for now.
	/*
	// Write the LaTeX header.
	fprintf(report->file, "\\begin{verbatim}\n");
	fflush(report->file);

	// Fork a child to copy the hardware file into the report file.
	// This is a bit of a hack, combining open and fopen...
	child_pid = fork();
	if ( child_pid == -1 ) {
		return exception_throw("Unable to fork cat child.", function_name);
	}
	else if ( child_pid == 0 ) {
		char report_tex[REPORT_FILENAME_LENGTH_MAX + 4];

		// Build the report filename.
		strcpy(report_tex, report->filename);
		strcat(report_tex, ".tex");

		// Open the report file.
		fd = open(report_tex, O_APPEND | O_SYNC | O_WRONLY);
		if ( fd == -1 ) {
			return exception_throw("Unable to open report file.", function_name);
		}

		// Close stdout.
		if ( close(STDOUT_FILENO) == -1 ) {
			return exception_throw("Unable to close cat stdout.", function_name);
		}

		// Close stderr.
		if ( close(STDERR_FILENO) == -1 ) {
			return exception_throw("Unable to close cat stderr.", function_name);
		}

		// Duplicate stdout to the report file.
		if ( dup2(fd, STDOUT_FILENO) == -1 ) {
			return exception_throw("Unable to dup2 to stdout.", function_name);
		}

		// Duplicate stderr to the report file.
		if ( dup2(fd, STDERR_FILENO) == -1 ) {
			return exception_throw("Unable to dup2 to stderr.", function_name);
		}

		// Close the report file descriptor.
		if ( close(fd) == -1 ) {
			return exception_throw("Unable to close report fd.", function_name);
		}

		// Execute cat.
		execlp("cat", "cat", filename, (char*)'\0');
		return exception_throw("Unable to exec into cat.", function_name);
	}

	// Wait for the child.
	if ( waitpid(child_pid, &child_status, 0) == -1 ) {
		return exception_throw("Unable to wait for child.", function_name);
	}

	// Check the child exit status.
	if ( WIFEXITED(child_status) ) {
		int exit_status;
		exit_status = WEXITSTATUS(child_status);
		if ( exit_status == EXIT_FAILURE ) {
			return exception_throw("Child returned EXIT_FAILURE.", function_name);
		}
	}
	else { 
		return exception_throw("Child did not exit properly.", function_name);
	}

	// Update the file descriptor since cat added data.
	fseek(report->file, 0, SEEK_END);

	// Write the LaTeX tail.
	fprintf(report->file, "\\end{verbatim}\n");
	*/

	// Return success.
	return NULL;
}
