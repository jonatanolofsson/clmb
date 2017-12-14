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
\   " -I" . s:TOP .
\   " -I" . s:TOP . "/libs/rapidjson/include" .
\   " -I" . s:TOP . "/libs/swss" .
\   " -isystem/usr/include/eigen3" .
\   " -fopenmp"

let g:ale_cpp_clang_options = s:cpp_options
let g:ale_cpp_clangcheck_options = s:cpp_options
let g:ale_cpp_clangtidy_options = s:cpp_options
let g:ale_cpp_cppcheck_options = s:cpp_options
let g:ale_cpp_cpplint_options = s:cpp_options
let g:ale_cpp_gcc_options = s:cpp_options
