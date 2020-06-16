Yee_2D

This program will solve Maxwell's equations using FDTD for a 2D space with a number of prid points in the x and y directions. It can run in three running modes; Serial (Standard), OpenMP, and OpenCL.

Original program - https://dougneubauer.com/wp-content/uploads/wdata/taflove2d/yee2d_c.txt by Dr. Susan C. Hagness

Parallelism implemented by CSC3002 student

Run with ./yee2d.x x=500 y=500 OCL

Run with ./yee2d.x x=500 y=500 STD DebugMode to enable the output of the field values to disk

When there is output for each running mode at the same X and Y values the validation.py script can be used to compare the different set of results.
The Validation.py may need modified starting (CurrentSize) and max (MaxSize) values in order to run correctly.
The Validation.py requires no passed values and should already be in the correct location, it can be run like a normal python script.
