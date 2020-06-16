#!/bin/make
#---+----1----+----2----+----3----+----4----+----5----+----6----+---7
#
#  Makefile: An OpenCL/OpenMP example program  
#
#---+----1----+----2----+----3----+----4----+----5----+----6----+---7

NVCC = nvcc -Xcompiler -fopenmp

RM = /bin/rm 

PROJ_DIR = $(PWD)

INCS=-I$(PROJ_DIR)/inc  
#   -I$(PROJ_DIR)/OutputLibraries/gifenc-master/inc

LIBS=-I$(PROJ_DIR)/lib

CXXFLAGS= -v -g -x cu

LINKFLAGS=-lOpenCL -lgomp -lm

#

SRCS_FDTD = $(PROJ_DIR)/src/yee2d.cl  

EXEC_FDTD = $(PROJ_DIR)/bin/yee2d.x

#

SRCS_GETDEV = $(PROJ_DIR)/src/getdevinfo.cl  

EXEC_GETDEV = $(PROJ_DIR)/bin/getdevinfo.x

#

all: clean getdevinfo fdtd

fdtd:
	$(NVCC) $(INCS) $(CXXFLAGS)  $(LIBS) -o $(EXEC_FDTD) \
                $(SRCS_FDTD) $(LINKFLAGS) 

getdevinfo:
	$(NVCC) $(INCS) $(CXXFLAGS)  $(LIBS) -o $(EXEC_GETDEV) \
                $(SRCS_GETDEV) $(LINKFLAGS) 

clean:
	$(RM) -vf $(PROJ_DIR)/src/*.o $(PROJ_DIR)/bin/*.x 

#---+----1----+----2----+----3----+----4----+----5----+----6----+---7
#
#  End Makefile 
#
#---+----1----+----2----+----3----+----4----+----5----+----6----+---7
