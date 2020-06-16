#********************************************************************
#******************************************************************** 
#
#   Makefile to build the executable(s)
#
#********************************************************************
#******************************************************************** 

CC = gcc

CCOPTS = -fopenmp -Wall -o3 -g --std=c99

RM = /bin/rm 

PROJ_DIR = $(PWD)

SRCS = $(PROJ_DIR)/src/fdtd3d_float.c 

EXEC = $(PROJ_DIR)/bin/fdtd3d.x

all: clean complink

complink:
	$(CC) $(CCOPTS) -o $(EXEC) $(SRCS) -lm 

clean:
	$(RM) -vf $(PROJ_DIR)/src/*.o $(PROJ_DIR)/bin/*.x 

