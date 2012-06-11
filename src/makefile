ccompiler = gcc
cucompiler = nvcc
linker = gcc
cflags = -fopenmp --pedantic-errors -Wall -Werror -std=c99
libraries = -lccc -lcuda -lcudart -lgomp

cfiles =	main \
		algorithm \
		benchmark \
		benchmark_data \
		block128 \
		encryption \
		file \
		key \
		mirror_bytes \
		mode \
		report \
		section \
		statistics \
		subsection \
		serpent \
		twofish
cufiles = 	cuda_extension \
		serpent_cu \
		twofish_cu # Linker does not like files with the same name.
		

# Version information.
major_number = 1
minor_number = 0
release_number = 0
version = ${major_number}.${minor_number}.${release_number}

# Binary name.
binary = benchmark

# Perform default functionality.
default: compile link tidy done

# Clean the directory.
clean: tidy
	@echo "  Cleaning."
	@rm -f ${binary}
	@rm -f ${script}
	@rm -f VERSION
	@rm -f libccc.so.0
	@echo "  Done."

# Compile the benchmark.
compile: compile_c compile_cu compile_ccc

# Compile C files.
compile_c:
	@for file in ${cfiles}; do \
		echo "  Compiling $$file.c."; \
		${ccompiler} ${cflags} -c $$file.c; \
	done

# Compile CUDA files.
compile_cu:
	@for file in ${cufiles}; do \
		echo "  Compiling $$file.cu."; \
		${cucompiler} -c $$file.cu; \
	done

# Compile Cline's C Compendium (CCC).
compile_ccc:
	@# This is a bit of a hack; copy and rename the shared object library
	@# to the local directory rather than installing it.
	@cd ccc; make; mv libccc.so* ../libccc.so; cd ../; chmod 0600 libccc.so

# Print the conclusion.
done:
	@echo "  Done."

# Link the executable.
link: link_binary link_hack link_version

# Create the binary file.
link_binary:
	@echo "  Creating ${binary}."
	@${linker} -I/usr/local/cuda/include/ -L./ -L/usr/local/cuda/lib64/ -L/usr/local/cuda/lib/ ${libraries} -o ${binary} *.o

# Hack because gcc either complains about no -lccc, or it complains about no libccc.so.0! Yeesh.
link_hack:
	@mv libccc.so libccc.so.0

# Output version information.
link_version:
	@echo "  Saving version."
	@echo ${version} > VERSION

# Clean excess files without removing important compilation output.
tidy:
	@echo "  Tidying."
	@rm -f *.o
