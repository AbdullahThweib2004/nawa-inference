// The one translation unit that compiles stb_image's implementation.
//
// stb_image is a single-header library: every file may include the header for the
// declarations, and exactly one file defines STB_IMAGE_IMPLEMENTATION to get the code.
// It is third-party code, so this file is built in its own target with warnings disabled
// (see the root CMakeLists.txt); everything else keeps -Wall -Wextra -Wpedantic -Werror.
//
// Image decoding is not math: it doesn't conflict with the "no external math libraries"
// rule in CLAUDE.md.
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
