CXX = g++
CXXFLAGS = -Wall -O2

LAPACK_FLAGS = -l lapack -l blas

TARGETS = problem_1 problem_2 dgeqrf problem_4

all: $(TARGETS)

problem_1: problem_1.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^

problem_2: problem_2.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^

problem_4: problem_4.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LAPACK_FLAGS)

dgeqrf: dgeqrf.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LAPACK_FLAGS)

clean:
	rm $(TARGETS)
