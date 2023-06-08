

blt_add_executable(
NAME sparseEval.exe
SOURCES main.cpp
  DEPENDS_ON RAJA openmp
)
