export CC = gcc
export CXX = g++
export CFLAGS = -w -O3 -pthread -std=c++0x

LDFLAGS = -lm -lgsl -lgslcblas -larmadillo -fpermissive
INSTALL_PATH=bin/
BIN = ctl
OBJ = ctl-estimate.o ctl-inference.o ctl-learn.o utils.o gsl-wrappers.o params.o
.PHONY: clean all

all: $(BIN)

ctl:ctl.cc ctl-estimate.o ctl-inference.o ctl-learn.o utils.o ctl.h gsl-wrappers.o params.o
ctl-estimate.o:ctl-estimate.cc utils.h ctl.h ctl-estimate.h
ctl-inference.o: ctl-inference.cc utils.h ctl-inference.h ctl.h
ctl-learn.o: ctl-learn.cc utils.h ctl-learn.h ctl.h
utils.o: utils.cc
gsl-wrappers.o: gsl-wrappers.cc
params.o: params.cc

$(BIN) : 
	$(CXX) $(filter %.cpp %.o %.cc, $^) $(CFLAGS) -o $@ $(LDFLAGS)

$(OBJ) :
	$(CXX) -c $(CFLAGS) -o $@ $(filter %.cpp %.cc, $^) $(LDFLAGS)

install:
	cp -f -r $(BIN) $(INSTALL_PATH)

clean:
	$(RM) $(OBJ) $(BIN) *~
