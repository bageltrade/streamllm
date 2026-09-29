// Termux / older libc++: headers may expect exported __hash_memory
// that the linked libc++ does not provide. Force the inline path.
#pragma once
#if defined(__ANDROID__) || defined(__TERMUX__) || defined(__linux__)
  #ifndef _LIBCPP_AVAILABILITY_HAS_HASH_MEMORY
    #define _LIBCPP_AVAILABILITY_HAS_HASH_MEMORY 0
  #endif
  // Older name used in some libc++ builds
  #ifndef _LIBCPP_AVAILABILITY_HAS_HASH
    #define _LIBCPP_AVAILABILITY_HAS_HASH 0
  #endif
#endif
