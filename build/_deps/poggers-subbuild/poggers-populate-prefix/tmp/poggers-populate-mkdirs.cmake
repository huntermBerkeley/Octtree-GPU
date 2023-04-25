# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/home/hunter/work/old_oct/Octtree-GPU/downloaded_libraries/poggers/6c190b04dd3b6316da6a18ab9e6032f078577d7b"
  "/home/hunter/work/old_oct/Octtree-GPU/build/_deps/poggers-build"
  "/home/hunter/work/old_oct/Octtree-GPU/build/_deps/poggers-subbuild/poggers-populate-prefix"
  "/home/hunter/work/old_oct/Octtree-GPU/build/_deps/poggers-subbuild/poggers-populate-prefix/tmp"
  "/home/hunter/work/old_oct/Octtree-GPU/build/_deps/poggers-subbuild/poggers-populate-prefix/src/poggers-populate-stamp"
  "/home/hunter/work/old_oct/Octtree-GPU/build/_deps/poggers-subbuild/poggers-populate-prefix/src"
  "/home/hunter/work/old_oct/Octtree-GPU/build/_deps/poggers-subbuild/poggers-populate-prefix/src/poggers-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/hunter/work/old_oct/Octtree-GPU/build/_deps/poggers-subbuild/poggers-populate-prefix/src/poggers-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/hunter/work/old_oct/Octtree-GPU/build/_deps/poggers-subbuild/poggers-populate-prefix/src/poggers-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
