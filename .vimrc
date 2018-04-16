let g:TOP = expand("<sfile>:p:h")
let s:includes =
\   " -I" . g:TOP . "/include" .
\   " -I" . g:TOP . "/libs/pybind11/include -I/usr/include/python3.6m -I/usr/include/python3.6m"
let s:cpp_options =
\   " -std=c++17" .
\   " -Wall" .
\   " -Werror" .
\   " -Wno-unknown-pragmas" .
\   " -pedantic-errors" .
\   " -Wextra" .
\   " -Wcast-align" .
\   " -O3" .
\   " -isystem/usr/include/eigen3" .
\   " -fopenmp"

let g:ale_cpp_gcc_options = s:includes . s:cpp_options
let g:ale_cpp_clang_options = s:includes . s:cpp_options
let g:ale_cpp_clangcheck_options = " --" . s:includes . s:cpp_options
let g:ale_cpp_clangtidy_options = " --" . s:includes . s:cpp_options
let g:ale_cpp_cppcheck_options = s:includes . " --std=c++11"
