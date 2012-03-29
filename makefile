ccompiler = gcc
cucompiler = nvcc
linker = gcc
cflags = -fopenmp --pedantic-errors -Wall -Werror -std=c99
libraries = -lccc -lcuda -lcudart -lgomp

cfiles =	main \
		block128 \
		file \
		mirror_bytes \
		serpent
cufiles = 	cuda_extension \
		serpent_cu # Linker does not like files with the same name.
		

# Version information.
major_number = 0
minor_number = 0
release_number = 1
version = ${major_number}.${minor_number}.${release_number}

# Benchmark name.
name = benchmark

# Perform default functionality.
default: compile libccc.so create clean done


# Compile the library (ccc).
libccc.so:
	@cd ccc; make; cd ../
	@mv ccc/libccc.so.0.0.1 ./libccc.so

# Clean the directory.
clean:
	@echo "  Cleaning."
	@rm -f *.o


# Compile the benchmark.
compile:
	@for file in ${cfiles}; do \
		echo "  Compiling $$file.c."; \
		${ccompiler} ${cflags} -c $$file.c; \
	done
	@for file in ${cufiles}; do \
		echo "  Compiling $$file.cu."; \
		${cucompiler} -c $$file.cu; \
	done


# Create the executable.
create:
	@echo "  Creating ${name}."
	@${linker} -I/usr/local/cuda/include/ -L./ -L/usr/local/cuda/lib/ ${libraries} -o ${name} *.o


# Print the conclusion.
done:
	@echo "  Done."

