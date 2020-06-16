#!/usr/bin/python

Increment = 100
CurrentSize = 100
MaxSize = 500

OutputFolder = "Output"
StandardFolder = OutputFolder + "/Standard"
OpenMPFolder = OutputFolder + "/OpenMP"
OpenCLFolder = OutputFolder + "/OpenCL"

while CurrentSize <= MaxSize:
    print(str(CurrentSize) + "\n")
    # HZ
    print("HZ\n")
    hzStandard = open(StandardFolder + "/" + str(CurrentSize)+ "_" + str(CurrentSize) + "_hz.txt", "r")
    hzOpenMP = open(OpenMPFolder + "/" + str(CurrentSize) + "_" + str(CurrentSize) + "_hz.txt", "r")
    hzOpenCL = open(OpenCLFolder + "/" + str(CurrentSize) + "_" + str(CurrentSize) + "_hz.txt", "r")

    sfLowestOpenMPhz = -1
    sfLowestOpenCLhz = -1
    
    for line in hzStandard:
        OpenMPLine = hzOpenMP.readline()
        OpenCLLine = hzOpenCL.readline()
        index = 0
        sfOpenMP = -1
        sfOpenCL = -1

        OMPValid = True
        OCLValid = True

        while (OMPValid==True or OCLValid==True) and index < len(line) and index < len(OpenMPLine) and index < len(OpenCLLine):
            if OMPValid==True:
                if line[index] == OpenMPLine[index]:
                    if line[index] != "-" and line[index] != "." and line[index] != " ":
                        sfOpenMP += 1
                else:
                    OCLValid == False

            if OCLValid==True:
                if line[index] == OpenCLLine[index]:
                    if line[index] != '-' and line[index] != '.' and line[index] != " ":
                        sfOpenCL += 1
                else:
                    OCLValid == False

            index += 1
        
        if sfLowestOpenMPhz == -1:
            sfLowestOpenMPhz = sfOpenMP
        elif sfOpenMP < sfLowestOpenMPhz:
            sfLowestOpenMPhz = sfOpenMP

        if sfLowestOpenCLhz == -1:
            sfLowestOpenCLhz = sfOpenCL
        elif sfOpenCL < sfLowestOpenCLhz:
            sfLowestOpenCLhz = sfOpenCL

    hzStandard.close()
    hzOpenMP.close()
    hzOpenCL.close()

    # HZ END

    # EX
    print("EX\n")
    exStandard = open(StandardFolder + "/" + str(CurrentSize) + "_" + str(CurrentSize) + "_ex.txt", "r")
    exOpenMP = open(OpenMPFolder + "/" + str(CurrentSize) + "_" + str(CurrentSize) + "_ex.txt", "r")
    exOpenCL = open(OpenCLFolder + "/" + str(CurrentSize) + "_" + str(CurrentSize) + "_ex.txt", "r")

    sfLowestOpenMPex = -1
    sfLowestOpenCLex = -1
    
    for line in exStandard:
        OpenMPLine = exOpenMP.readline()
        OpenCLLine = exOpenCL.readline()
        index = 0
        sfOpenMP = -1
        sfOpenCL = -1

        OMPValid = True
        OCLValid = True

        while (OMPValid==True or OCLValid==True) and index < len(line) and index < len(OpenMPLine) and index < len(OpenCLLine):
            if OMPValid==True:
                if line[index] == OpenMPLine[index]:
                    if line[index] != "-" and line[index] != "." and line[index] != " ":
                        sfOpenMP += 1
                else:
                    OCLValid == False

            if OCLValid==True:
                if line[index] == OpenCLLine[index]:
                    if line[index] != "-" and line[index] != "." and line[index] != " ":
                        sfOpenCL += 1
                else:
                    OCLValid == False

            index += 1

        if sfLowestOpenMPex == -1:
            sfLowestOpenMPex = sfOpenMP
        elif sfOpenMP < sfLowestOpenMPex:
            sfLowestOpenMPex = sfOpenMP

        if sfLowestOpenCLex == -1:
            sfLowestOpenCLex = sfOpenCL
        elif sfOpenCL < sfLowestOpenCLex:
            sfLowestOpenCLex = sfOpenCL


    exStandard.close()
    exOpenMP.close()
    exOpenCL.close()
    # EX END

    # EY
    print("EY\n")
    eyStandard = open(StandardFolder + "/" + str(CurrentSize) + "_" + str(CurrentSize) + "_ey.txt", "r")
    eyOpenMP = open(OpenMPFolder + "/" + str(CurrentSize) + "_" + str(CurrentSize) + "_ey.txt", "r")
    eyOpenCL = open(OpenCLFolder + "/" + str(CurrentSize) + "_" + str(CurrentSize) + "_ey.txt", "r")

    sfLowestOpenMPey = -1
    sfLowestOpenCLey = -1
    
    for line in eyStandard:
        OpenMPLine = eyOpenMP.readline()
        OpenCLLine = eyOpenCL.readline()
        index = 0
        sfOpenMP = -1
        sfOpenCL = -1

        OMPValid = True
        OCLValid = True

        while (OMPValid==True or OCLValid==True) and index < len(line) and index < len(OpenMPLine) and index < len(OpenCLLine):
            if OMPValid==True:
                if line[index] == OpenMPLine[index]:
                    if line[index] != "-" and line[index] != "." and line[index] != " ":
                        sfOpenMP += 1
                else:
                    OCLValid == False

            if OCLValid==True:
                if line[index] == OpenCLLine[index]:
                    if line[index] != "-" and line[index] != "." and line[index] != " ":
                        sfOpenCL += 1
                else:
                    OCLValid == False

            index += 1
        
        if sfLowestOpenMPey == -1:
            sfLowestOpenMPey = sfOpenMP
        elif sfOpenMP < sfLowestOpenMPey:
            sfLowestOpenMPey = sfOpenMP

        if sfLowestOpenCLey == -1:
            sfLowestOpenCLey = sfOpenCL
        elif sfOpenCL < sfLowestOpenCLey:
            sfLowestOpenCLey = sfOpenCL


    exStandard.close()
    exOpenMP.close()
    exOpenCL.close()

    # EY END
    print("Lowest correct significant figure for OpenMP ex: " + str(sfLowestOpenMPex) + "\n")
    print("Lowest correct significant figure for OpenMP ey: " + str(sfLowestOpenMPey) + "\n")
    print("Lowest correct significant figure for OpenMP hz: " + str(sfLowestOpenMPhz) + "\n")
    print("Lowest correct significant figure for OpenCL ex: " + str(sfLowestOpenCLex) + "\n")
    print("Lowest correct significant figure for OpenCL ey: " + str(sfLowestOpenCLey) + "\n")
    print("Lowest correct significant figure for OpenCL hz: " + str(sfLowestOpenCLhz) + "\n")

    f = open(OutputFolder + "/" + str(CurrentSize ) + "_" + str(CurrentSize ) + "_validation.txt", 'w+')
    f.write("Lowest correct significant figure for OpenMP ex: " + str(sfLowestOpenMPex) + "\n")
    f.write("Lowest correct significant figure for OpenMP ey: " + str(sfLowestOpenMPey) + "\n")
    f.write("Lowest correct significant figure for OpenMP hz: " + str(sfLowestOpenMPhz) + "\n")
    f.write("Lowest correct significant figure for OpenCL ex: " + str(sfLowestOpenCLex) + "\n")
    f.write("Lowest correct significant figure for OpenCL ey: " + str(sfLowestOpenCLey) + "\n")
    f.write("Lowest correct significant figure for OpenCL hz: " + str(sfLowestOpenCLhz) + "\n")
    f.close()

    CurrentSize = CurrentSize + Increment

