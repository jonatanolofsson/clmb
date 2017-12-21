#pragma once

#ifdef NOPAR
    #define PARFOR
#else
    #define PARFOR _Pragma("omp parallel for")
#endif

