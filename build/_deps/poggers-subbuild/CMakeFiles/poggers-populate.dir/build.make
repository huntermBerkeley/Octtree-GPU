# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.26

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/hunter/work/old_oct/Octtree-GPU/build/_deps/poggers-subbuild

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/hunter/work/old_oct/Octtree-GPU/build/_deps/poggers-subbuild

# Utility rule file for poggers-populate.

# Include any custom commands dependencies for this target.
include CMakeFiles/poggers-populate.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/poggers-populate.dir/progress.make

CMakeFiles/poggers-populate: CMakeFiles/poggers-populate-complete

CMakeFiles/poggers-populate-complete: poggers-populate-prefix/src/poggers-populate-stamp/poggers-populate-install
CMakeFiles/poggers-populate-complete: poggers-populate-prefix/src/poggers-populate-stamp/poggers-populate-mkdir
CMakeFiles/poggers-populate-complete: poggers-populate-prefix/src/poggers-populate-stamp/poggers-populate-download
CMakeFiles/poggers-populate-complete: poggers-populate-prefix/src/poggers-populate-stamp/poggers-populate-update
CMakeFiles/poggers-populate-complete: poggers-populate-prefix/src/poggers-populate-stamp/poggers-populate-patch
CMakeFiles/poggers-populate-complete: poggers-populate-prefix/src/poggers-populate-stamp/poggers-populate-configure
CMakeFiles/poggers-populate-complete: poggers-populate-prefix/src/poggers-populate-stamp/poggers-populate-build
CMakeFiles/poggers-populate-complete: poggers-populate-prefix/src/poggers-populate-stamp/poggers-populate-install
CMakeFiles/poggers-populate-complete: poggers-populate-prefix/src/poggers-populate-stamp/poggers-populate-test
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/hunter/work/old_oct/Octtree-GPU/build/_deps/poggers-subbuild/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Completed 'poggers-populate'"
	/usr/bin/cmake -E make_directory /home/hunter/work/old_oct/Octtree-GPU/build/_deps/poggers-subbuild/CMakeFiles
	/usr/bin/cmake -E touch /home/hunter/work/old_oct/Octtree-GPU/build/_deps/poggers-subbuild/CMakeFiles/poggers-populate-complete
	/usr/bin/cmake -E touch /home/hunter/work/old_oct/Octtree-GPU/build/_deps/poggers-subbuild/poggers-populate-prefix/src/poggers-populate-stamp/poggers-populate-done

poggers-populate-prefix/src/poggers-populate-stamp/poggers-populate-update:
.PHONY : poggers-populate-prefix/src/poggers-populate-stamp/poggers-populate-update

poggers-populate-prefix/src/poggers-populate-stamp/poggers-populate-build: poggers-populate-prefix/src/poggers-populate-stamp/poggers-populate-configure
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/hunter/work/old_oct/Octtree-GPU/build/_deps/poggers-subbuild/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "No build step for 'poggers-populate'"
	cd /home/hunter/work/old_oct/Octtree-GPU/build/_deps/poggers-build && /usr/bin/cmake -E echo_append
	cd /home/hunter/work/old_oct/Octtree-GPU/build/_deps/poggers-build && /usr/bin/cmake -E touch /home/hunter/work/old_oct/Octtree-GPU/build/_deps/poggers-subbuild/poggers-populate-prefix/src/poggers-populate-stamp/poggers-populate-build

poggers-populate-prefix/src/poggers-populate-stamp/poggers-populate-configure: poggers-populate-prefix/tmp/poggers-populate-cfgcmd.txt
poggers-populate-prefix/src/poggers-populate-stamp/poggers-populate-configure: poggers-populate-prefix/src/poggers-populate-stamp/poggers-populate-patch
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/hunter/work/old_oct/Octtree-GPU/build/_deps/poggers-subbuild/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "No configure step for 'poggers-populate'"
	cd /home/hunter/work/old_oct/Octtree-GPU/build/_deps/poggers-build && /usr/bin/cmake -E echo_append
	cd /home/hunter/work/old_oct/Octtree-GPU/build/_deps/poggers-build && /usr/bin/cmake -E touch /home/hunter/work/old_oct/Octtree-GPU/build/_deps/poggers-subbuild/poggers-populate-prefix/src/poggers-populate-stamp/poggers-populate-configure

poggers-populate-prefix/src/poggers-populate-stamp/poggers-populate-download: poggers-populate-prefix/src/poggers-populate-stamp/poggers-populate-gitinfo.txt
poggers-populate-prefix/src/poggers-populate-stamp/poggers-populate-download: poggers-populate-prefix/src/poggers-populate-stamp/poggers-populate-mkdir
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/hunter/work/old_oct/Octtree-GPU/build/_deps/poggers-subbuild/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Performing download step (git clone) for 'poggers-populate'"
	cd /home/hunter/work/old_oct/Octtree-GPU/downloaded_libraries/poggers && /usr/bin/cmake -P /home/hunter/work/old_oct/Octtree-GPU/build/_deps/poggers-subbuild/poggers-populate-prefix/tmp/poggers-populate-gitclone.cmake
	cd /home/hunter/work/old_oct/Octtree-GPU/downloaded_libraries/poggers && /usr/bin/cmake -E touch /home/hunter/work/old_oct/Octtree-GPU/build/_deps/poggers-subbuild/poggers-populate-prefix/src/poggers-populate-stamp/poggers-populate-download

poggers-populate-prefix/src/poggers-populate-stamp/poggers-populate-install: poggers-populate-prefix/src/poggers-populate-stamp/poggers-populate-build
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/hunter/work/old_oct/Octtree-GPU/build/_deps/poggers-subbuild/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "No install step for 'poggers-populate'"
	cd /home/hunter/work/old_oct/Octtree-GPU/build/_deps/poggers-build && /usr/bin/cmake -E echo_append
	cd /home/hunter/work/old_oct/Octtree-GPU/build/_deps/poggers-build && /usr/bin/cmake -E touch /home/hunter/work/old_oct/Octtree-GPU/build/_deps/poggers-subbuild/poggers-populate-prefix/src/poggers-populate-stamp/poggers-populate-install

poggers-populate-prefix/src/poggers-populate-stamp/poggers-populate-mkdir:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/hunter/work/old_oct/Octtree-GPU/build/_deps/poggers-subbuild/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Creating directories for 'poggers-populate'"
	/usr/bin/cmake -Dcfgdir= -P /home/hunter/work/old_oct/Octtree-GPU/build/_deps/poggers-subbuild/poggers-populate-prefix/tmp/poggers-populate-mkdirs.cmake
	/usr/bin/cmake -E touch /home/hunter/work/old_oct/Octtree-GPU/build/_deps/poggers-subbuild/poggers-populate-prefix/src/poggers-populate-stamp/poggers-populate-mkdir

poggers-populate-prefix/src/poggers-populate-stamp/poggers-populate-patch: poggers-populate-prefix/src/poggers-populate-stamp/poggers-populate-update
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/hunter/work/old_oct/Octtree-GPU/build/_deps/poggers-subbuild/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "No patch step for 'poggers-populate'"
	/usr/bin/cmake -E echo_append
	/usr/bin/cmake -E touch /home/hunter/work/old_oct/Octtree-GPU/build/_deps/poggers-subbuild/poggers-populate-prefix/src/poggers-populate-stamp/poggers-populate-patch

poggers-populate-prefix/src/poggers-populate-stamp/poggers-populate-update:
.PHONY : poggers-populate-prefix/src/poggers-populate-stamp/poggers-populate-update

poggers-populate-prefix/src/poggers-populate-stamp/poggers-populate-test: poggers-populate-prefix/src/poggers-populate-stamp/poggers-populate-install
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/hunter/work/old_oct/Octtree-GPU/build/_deps/poggers-subbuild/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "No test step for 'poggers-populate'"
	cd /home/hunter/work/old_oct/Octtree-GPU/build/_deps/poggers-build && /usr/bin/cmake -E echo_append
	cd /home/hunter/work/old_oct/Octtree-GPU/build/_deps/poggers-build && /usr/bin/cmake -E touch /home/hunter/work/old_oct/Octtree-GPU/build/_deps/poggers-subbuild/poggers-populate-prefix/src/poggers-populate-stamp/poggers-populate-test

poggers-populate-prefix/src/poggers-populate-stamp/poggers-populate-update: poggers-populate-prefix/src/poggers-populate-stamp/poggers-populate-download
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/hunter/work/old_oct/Octtree-GPU/build/_deps/poggers-subbuild/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Performing update step for 'poggers-populate'"
	cd /home/hunter/work/old_oct/Octtree-GPU/downloaded_libraries/poggers/6c190b04dd3b6316da6a18ab9e6032f078577d7b && /usr/bin/cmake -P /home/hunter/work/old_oct/Octtree-GPU/build/_deps/poggers-subbuild/poggers-populate-prefix/tmp/poggers-populate-gitupdate.cmake

poggers-populate: CMakeFiles/poggers-populate
poggers-populate: CMakeFiles/poggers-populate-complete
poggers-populate: poggers-populate-prefix/src/poggers-populate-stamp/poggers-populate-build
poggers-populate: poggers-populate-prefix/src/poggers-populate-stamp/poggers-populate-configure
poggers-populate: poggers-populate-prefix/src/poggers-populate-stamp/poggers-populate-download
poggers-populate: poggers-populate-prefix/src/poggers-populate-stamp/poggers-populate-install
poggers-populate: poggers-populate-prefix/src/poggers-populate-stamp/poggers-populate-mkdir
poggers-populate: poggers-populate-prefix/src/poggers-populate-stamp/poggers-populate-patch
poggers-populate: poggers-populate-prefix/src/poggers-populate-stamp/poggers-populate-test
poggers-populate: poggers-populate-prefix/src/poggers-populate-stamp/poggers-populate-update
poggers-populate: CMakeFiles/poggers-populate.dir/build.make
.PHONY : poggers-populate

# Rule to build all files generated by this target.
CMakeFiles/poggers-populate.dir/build: poggers-populate
.PHONY : CMakeFiles/poggers-populate.dir/build

CMakeFiles/poggers-populate.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/poggers-populate.dir/cmake_clean.cmake
.PHONY : CMakeFiles/poggers-populate.dir/clean

CMakeFiles/poggers-populate.dir/depend:
	cd /home/hunter/work/old_oct/Octtree-GPU/build/_deps/poggers-subbuild && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/hunter/work/old_oct/Octtree-GPU/build/_deps/poggers-subbuild /home/hunter/work/old_oct/Octtree-GPU/build/_deps/poggers-subbuild /home/hunter/work/old_oct/Octtree-GPU/build/_deps/poggers-subbuild /home/hunter/work/old_oct/Octtree-GPU/build/_deps/poggers-subbuild /home/hunter/work/old_oct/Octtree-GPU/build/_deps/poggers-subbuild/CMakeFiles/poggers-populate.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/poggers-populate.dir/depend

