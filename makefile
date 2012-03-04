compiler = gcc
flags = --pedantic-errors -Wall -Werror -std=c99
libraries = -lccc

files =	main \
	block128 \
	file \
	mirror_bytes \
	serpent

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
	@for file in ${files}; do \
		echo "  Compiling $$file.o."; \
		${compiler} ${flags} -c $$file.c; \
	done


# Create the executable.
create:
	@echo "  Creating ${name}."
	@${compiler} -L./ ${libraries} -o ${name} *.o


# Print the conclusion.
done:
	@echo "  Done."

