// Host-build stub. Heap caps degrade to plain malloc.
#pragma once
#include <cstdlib>
#define MALLOC_CAP_INTERNAL 0
#define MALLOC_CAP_8BIT 0
inline void* heap_caps_malloc(size_t n, int) { return malloc(n); }
