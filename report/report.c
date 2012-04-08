#include "report.h"


exception_t* report_init(report_t* report) {
	char* function_name = "report_init()";
	char filepath[REPORT_BASEPATH_LENGTH_MAX + REPORT_FILENAME_LENGTH_MAX + 3];
	exception_t* exception;

	// Validate parameters.
	if ( report == NULL ) {
		return exception_throw("report was NULL.", function_name);
	}

	// Set the report name.
	strcpy(report->filename, "report");

	// Create the directory structure.
	report->basepath[0] = '\0';
	exception = report_init_create_directories(report);
	if ( exception != NULL ) {
		return exception_append(exception, function_name);
	}
	
	// Open the report file.
	strcat(filepath, report->basepath);
	strcat(filepath, report->filename);
	strcat(filepath, ".tex");
	report->file = fopen(filepath, "w");
	if ( report->file == NULL ) {
		return exception_throw("Unable to create report file.", function_name);
	}

	// Initialize each section of the report.
	// Serpent section.
	exception = section_init(&(report->sections[REPORT_SECTION_SERPENT]), SERPENT);
	if ( exception != NULL ) {
		return exception_append(exception, function_name);
	}

	// Return success.
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
	strcpy(report->basepath, "results");
	if ( stat(report->basepath, &stats) == 0 ) { // File exists.
		// Check if the file is a directory.
		if ( !(S_ISDIR(stats.st_mode)) ) {
			return exception_throw("The results file exists and is not a directory.", function_name);
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

	// Invalidate the basepath.
	report->basepath[0] = '\0';

	// Invalidate the filename.
	report->filename[0] = '\0';

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
	return NULL;
}


exception_t* report_write(report_t* report) { 
	char* function_name = "report_write()";
	exception_t* exception;
	int i;

	// Validate parameters.
	if ( report == NULL ) {
		return exception_throw("report was NULL.", function_name);
	}

	// Write document head.
	fprintf(report->file, "\\documentclass{article}\n\n");
	fprintf(report->file, "\\usepackage{amsmath}\n\n");
	fprintf(report->file, "\\title{CUDA Benchmarking Report}\n\\author{Automatically Generated}\n\n");
	fprintf(report->file, "\\begin{document}\n\n");
	fprintf(report->file, "\\maketitle\n\n");

	// Write introduction section. TODOO
	fprintf(report->file, "\\section{Introduction}\n");

	// Write subsections.
	fprintf(report->file, "\n\\section{Results}\n");
	for ( i = 0; i < REPORT_SECTION_COUNT; i++ ) {
		exception = section_write(&(report->sections[i]), report->file);
		if ( exception != NULL ) {
			return exception_append(exception, function_name);
		}
	}

	// Write conclusion section. TODOO
	fprintf(report->file, "\n\\section{Conclusions}\n");
	
	// Write document tail.
	fprintf(report->file, "\\end{document}\n");

	// Flush the file buffer.
	if ( fflush(report->file) == EOF ) {
		return exception_throw("Unable to flush before write.", function_name);
	}

	// Compile the report into LaTeX.
	exception = report_write_compile_latex(report);
	if ( exception != NULL ) {
		return exception_append(exception, function_name);
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
	int i;

	// Move into the report directory.
	if ( chdir(report->basepath) == -1 ) {
		perror(NULL);
		return exception_throw("Unable to chdir into report directory.", function_name);
	}

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

	// Move back into the original directory.
	if ( chdir("../../") == -1 ) {
		perror(NULL);
		return exception_throw("Unable to chdir into original directory.", function_name);
	}

	// Return success.
	return NULL;
}


