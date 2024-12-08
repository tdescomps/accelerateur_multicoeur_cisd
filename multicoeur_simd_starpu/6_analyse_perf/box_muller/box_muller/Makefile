PRGS=box_muller

CXXFLAGS  = -Wall -O3
CXXFLAGS += -std=c++11
#CXXFLAGS += -march=native -mavx2
#CXXFLAGS += -qopt-report -qopenmp
#CXXFLAGS += -Ofast

CXXFLAGS_EXE = -g

.phony: all clean

all: $(PRGS)
clean:
	rm -fv $(PRGS) *.o *.s

%: %.cpp
	$(CXX) $(CXXFLAGS) $(CXXFLAGS_EXE) -o $@ $<

%.s: %.cpp
	$(CXX) $(CXXFLAGS) -S -o $@ $<
