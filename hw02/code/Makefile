CXX = g++
CXXFLAGS = -Wall -o2

LAPACK_FLAGS = -llapack -lblas

TARGETS = classical_gram_schmidt householder_qr dgeqrf

all: $(TARGETS)

# problem_5_6_7: problem_5_6_7.cpp
# 	$(CXX) $(CXXFLAGS) -o $@ $^ $(LAPACK_FLAGS)

classical_gram_schmidt: classical_gram_schmidt.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^

householder_qr: householder_qr.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^

dgeqrf: dgeqrf.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LAPACK_FLAGS)

clean:
	rm $(TARGETS)
