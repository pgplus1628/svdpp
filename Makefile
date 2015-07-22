CXX?=g++
CXXFLAGS?=-O3 -g -std=c++11
INCLUDES?=-I.
LDFLAGS?= -lglog -lgflags

svdpp : svdpp.o
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

svdpp.o : svdpp.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@


.PHONY:
clean : 
	rm -f svdpp.o
	rm -f svd
