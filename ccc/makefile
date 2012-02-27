compiler = gcc
flags = --pedantic-errors -Wall -Werror -fPIC -std=c99


# Library files.
files =	exception \
	math_extension

# Version information.
major_number = 0
minor_number = 0
release_number = 1
version = ${major_number}.${minor_number}.${release_number}

# Library name.
name = ccc
linker_name = lib${name}.so
soname = ${linker_name}.${major_number}
real_name = ${linker_name}.${version}

# Library paths.
base_path = /usr
include_path = ${base_path}/include
library_path = ${base_path}/lib


# Perform default functionality.
default: compile create clean done

# Install the library.
install: install_actual done


# Clean the directory.
clean:
	@echo "  Cleaning."
	@rm -f *.o


# Compile the library.
compile:
	@for file in ${files}; do \
		echo "  Compiling $$file.o."; \
		${compiler} ${flags} -c $$file.c; \
	done


# Create the library.
create:
	@echo "  Creating Cline's C Compendium."
	@${compiler} -shared -Wl,-soname,${soname} -o ${real_name} *.o


# Print the conclusion.
done:
	@echo "  Done."


# Install the library.
install_actual:
	@echo "  Installing."
	@# Install library files.
	@mv ${real_name} ${library_path}/
	@ln -sf ${real_name} ${library_path}/${linker_name}
	@ln -sf ${real_name} ${library_path}/${soname}
	@# Install header files.
	@mkdir -p ${include_path}/ccc
	@cp -f *.h ${include_path}/ccc/
	@chmod 0744 ${include_path}/ccc/*.h
