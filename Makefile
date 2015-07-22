CXX?=g++
CXXFLAGS?=-O3 -g -std=c++11
INCLUDES?=-I.
LDFLAGS?=-lgflags -lglog

svdpp : svdpp.o
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFALGS)

svdpp.o : svdpp.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@


.PHONY:
clean : 
	rm -f svdpp.o
	rm -f svd
