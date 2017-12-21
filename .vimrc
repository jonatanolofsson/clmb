let s:TOP = expand("<sfile>:p:h")
let s:cpp_options =
\   " -std=c++17" .
\   " -Wall" .
\   " -Werror" .
\   " -Wno-unknown-pragmas" .
\   " -Wfatal-errors" .
\   " -pedantic-errors" .
\   " -Wextra" .
\   " -Wcast-align" .
\   " -O3" .
\   " -I" . s:TOP . "/include" .
\   " -I" . s:TOP . "/libs/rapidjson/include" .
\   " -I" . s:TOP . "/libs/pybind11/include -I/usr/include/python3.6m -I/usr/include/python3.6m  -Wno-unused-result -Wsign-compare -march=x86-64 -mtune=generic -O2 -pipe -fstack-protector-strong -fno-plt -DNDEBUG -g -fwrapv -O3 -Wall -Wstrict-prototypes" .
\   " -I" . s:TOP . "/libs/swss" .
\   " -isystem/usr/include/eigen3" .
\   " -fopenmp"

let g:ale_cpp_clang_options = s:cpp_options
let g:ale_cpp_clangcheck_options = s:cpp_options
let g:ale_cpp_clangtidy_options = s:cpp_options
let g:ale_cpp_cppcheck_options = s:cpp_options
let g:ale_cpp_cpplint_options = s:cpp_options
let g:ale_cpp_gcc_options = s:cpp_options
