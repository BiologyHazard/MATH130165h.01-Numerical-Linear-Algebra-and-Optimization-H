CXX = g++
CXXFLAGS = -Wall -std=c++23

LAPACK_FLAGS = -llapack -lblas

TARGETS = problem_5_6_7

all: $(TARGETS)

problem_5_6_7: problem_5_6_7.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LAPACK_FLAGS)

clean:
	rm -f $(TARGETS)
