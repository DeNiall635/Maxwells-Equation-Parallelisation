
//***********************************************************************
//     2-D FDTD TE code with PML absorbing boundary conditions
//***********************************************************************
//     ported to C: Doug Neubauer, 4/02/2008
//
//     Matlab Program author: Susan C. Hagness
//                     Department of Electrical and Computer Engineering
//                     University of Wisconsin-Madison
//                     1415 Engineering Drive
//                     Madison, WI 53706-1691
//                     608-265-5739
//                     hagness@engr.wisc.edu
//
//     Date of this Matlab version:  February 2000
//
//     This C program (was Matlab) implements the finite-difference time-domain
//     solution of Maxwell's curl equations over a two-dimensional
//     Cartesian space lattice comprised of uniform square grid cells.
//
//     To illustrate the algorithm, a 6-cm-diameter metal cylindrical
//     scatterer in free space is modeled. The source excitation is
//     a Gaussian pulse with a carrier frequency of 5 GHz.
//
//     The grid resolution (dx = 3 mm) was chosen to provide 20 samples
//     per wavelength at the center frequency of the pulse (which in turn
//     provides approximately 10 samples per wavelength at the high end
//     of the excitation spectrum, around 10 GHz).
//
//     The computational domain is truncated using the perfectly matched
//     layer (PML) absorbing boundary conditions.  The formulation used
//     in this code is based on the original split-field Berenger PML. The
//     PML regions are labeled as shown in the following diagram:
//
//            ----------------------------------------------
//           |  |                BACK PML                |  |
//            ----------------------------------------------
//           |L |                                       /| R|
//           |E |                                (ib,jb) | I|
//           |F |                                        | G|
//           |T |                                        | H|
//           |  |                MAIN GRID               | T|
//           |P |                                        |  |
//           |M |                                        | P|
//           |L | (1,1)                                  | M|
//           |  |/                                       | L|
//            ----------------------------------------------
//           |  |                FRONT PML               |  |
//            ----------------------------------------------
//
// Important porting note:
//  indexes in Matlab start at 1, and in C they start at 0.
//  This can cause massive confusion when comparing the matlab .m file with the .c file
//
// In the spirit of the ToyFdtd programs, a point was made to try to heavily comment the source code.
//
// Below are results of a simple test: compare simulation result of hz[9][50]
// with ideal reference values and calculate average error over length of
// the simulation from 120 <= n <= 399
//
//     Test Results
//     --------------------------------------------------------------------
//     #layer   r0(rmax)    m(orderbc)     dB              comments
//     ------   --------    ----------    -------     ---------------------
//       8      0.00001%       2           -74.4        r0 1.0e-5% = 1.0e-7
//       8         "           3           -81.2
//       8         "           4           -81.1
//       8         "           5           -79.4
//       8         "           6           -77.7
//      10         "           3           -90.3
//      10      0.000001%      3           -88.5        r0 1.0e-6% = 1.0e-8
//      10      0.000005%      3           -89.7
//      10      0.00005%       3           -91.3
//      10      0.0001%        3           -91.7
//      10      0.0005%        3           -91.8
//      10      1.0e-14%       3           -81.7
//       6      1.0e-5%        3           -71.8
//       4      1.0e-5%        3           -51.8
//      pec:                                 0.0        For reference
//     mur1st:                             -30.6           "
//     liao:                               -30.4           "
//
//     conclusion: The PML ABC looks to be functioning properly
//
//***********************************************************************

// STANDARD LIBRARIES
#include <sys/types.h>
#include <sys/stat.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <string.h>
#include <math.h>
#include <omp.h>

// OPENCL LIBRARY
//#include </usr/include/CL/cl.h>  // AMD version 

#include <CL/opencl.h> // NVidia version 


// CUSTOM LIBRARIES
#include "yee2d.h"

// OUTPUT LIBRARIES
//#include "gifenc.h"

#define MEDIACONSTANT (2)
#define NUMBEROFITERATIONCONSTANT (500) // 
#define BIGLINESIZE (8192)
 
static  int nTotalCPUBytes = 0;  // Global and statically linked

static void addCPUCount(int const abytes, char const * const pclabel)
  {
   if(abytes > 0)
     {
      nTotalCPUBytes += abytes;

     //  printf("\n\n");

     //  printf("    ");

     //  if(NULL != pclabel) printf("%s ", pclabel);

     //  printf("Added %d bytes to total CPU memory allocation : Total CPUS allocation now: %d bytes",
     //         abytes, nTotalCPUBytes); 

     //  printf("\n\n");
     }

   return;
  }
 
static void printCPUCount() 
  {
   printf("     Total CPU memory allocation now: %d bytes", nTotalCPUBytes); 

   return;
  }


int main(int argc, char** argv)
{
    enum RunningMode runningMode;
    int xRegion, yRegion, i, debugMode;
    int size = 500;
    int help = 0;
    int runningModeDefined = 0;
    
    runningMode = Standard;
    xRegion = size, yRegion = size;
    debugMode = 0;

    printf("\n\n");
    printf("      2D FDTD program \n");
    printf("      ---------------   ");
    printf("\n\n");

    if (argc == 1)
    {
        printf("No Arguements givem, defaulting 500 * 500 2D space in standard running\n");
        EvaluateFdtd(runningMode, xRegion, yRegion, debugMode);
    }
    else if (argc >= 2)
    {
        for (i = 1; i < argc; i++)
        {
            char *var = argv[i];

            if (strcmp("help", var) == 0)
            {
                printf("Enter variables for the number of grid points in the x and y directiona with x={NUM} and y={NUM}, e.g. x=500\n");
                printf("Along with the Running Mode Standard / OpenMP / OpenCL\n");
                printf("For DebugMode enter DebugMode\n");
                help = 1;
            }
            else if (strcmp("standard", var) == 0 || strcmp("Standard", var) == 0 || strcmp("std", var) == 0 || strcmp("STD", var) == 0)
            {
                if (runningModeDefined == 0)
                {
                    runningMode = Standard;
                    printf("Standard Mode Enabled\n");
                    runningModeDefined = 1;
                }
            }
            else if (strcmp("openmp", var) == 0 || strcmp("OpenMP", var) == 0 || strcmp("omp", var) == 0 || strcmp("OMP", var) == 0)
            {
                if (runningModeDefined == 0)
                {
                    runningMode = OpenMP;
                    printf("OpenMP Mode Enabled\n");
                    runningModeDefined = 1;
                }
            }
            else if (strcmp("opencl", var) == 0 || strcmp("OpenCL", var) == 0 || strcmp("ocl", var) == 0 || strcmp("OCL", var) == 0)
            {
                if (runningModeDefined == 0)
                {
                    runningMode = OpenCL;
                    printf("     OpenCL Mode Enabled\n");
                    runningModeDefined = 1;
                }
            }
            else if (strcmp("DebugMode", var) == 0 || strcmp("debugmode", var) == 0)
            {
                debugMode = 1;
                printf("Debug Mode Enabled\n");
            }
            else if (var[0] == 'x')
            {
                char *x = var;
                x++;
                x++;
                xRegion = atoi(x);
                printf("     %d Set For the X Region\n", xRegion);
            }
            else if (var[0] == 'y')
            {
                char *y = var;
                y++;
                y++;
                yRegion = atoi(y);
                printf("     %d Set For the Y Region\n", yRegion);
            }
        }
        if (help != 1)
        {
            EvaluateFdtd(runningMode, xRegion, yRegion, debugMode);
        }
    }
    return (0);
}

void EvaluateFdtd(enum RunningMode runningMode, int xRegion, int yRegion, int debugMode)
{
    //***********************************************************************
    //     Printing/Plotting variables
    //***********************************************************************
    double minimumValue, maximumValue;
    minimumValue = -0.1;
    maximumValue = 0.1;

    int plottingInterval, centery, centerx;
    plottingInterval = 0;
    centery = 25;
    centerx = 15;

    //***********************************************************************
    //     Fundamental constants
    //***********************************************************************
    double cc, muz, epsz, pi;
    pi = (acos(-1.0));
    cc = 2.99792458e8;            //speed of light in free space (meters/second)
    muz = 4.0 * pi * 1.0e-7;      //permeability of free space
    epsz = 1.0 / (cc * cc * muz); //permittivity of free space

    double freq, lambda, omega;
    freq = 5.0e+9;           //center frequency of source excitation (Hz)
    lambda = cc / freq;      //center wavelength of source excitation
    omega = 2.0 * pi * freq; //center frequency in radians

    //***********************************************************************
    //     Grid parameters
    //***********************************************************************
    int ie, je;
    ie = xRegion; //number of grid cells in x-direction
    je = yRegion; //number of grid cells in y-direction

    int ib, jb;
    ib = ie + 1; // one extra is needed for fields on the boundaries (ie Ex on top boundary, Ey on right boundary)
    jb = je + 1; // ditto

    int  is, js;
    is = 15;     //location of z-directed hard source
    js = je / 2; //location of z-directed hard source

    double dx, dt;
    dx = 3.0e-3;          //space increment of square lattice  (meters)
    dt = dx / (2.0 * cc); //time step,  seconds, courant limit, Taflove1995 page 177

    int nmax;
    nmax = NUMBEROFITERATIONCONSTANT; //total number of time steps

    int iebc, jebc;
    iebc = 8;       //thickness of left and right PML region
    jebc = 8;       //thickness of front and back PML region

    double rmax, orderbc;
    rmax = 0.00001; // R(0) reflection coefficient (in %)  Nikolova part4 p.25
    orderbc = 2;    // m, grading order, optimal values: 2 <= m <= 6,  Nikolova part4 p.29

    int ibbc, jbbc, iefbc, jefbc, ibfbc, jbfbc;;
    ibbc = iebc + 1;
    jbbc = jebc + 1;
    iefbc = ie + 2 * iebc; // for front and bottom (width of region)
    jefbc = je + 2 * jebc; // not used
    ibfbc = iefbc + 1;     // one extra for Ey on right boundary
    jbfbc = jefbc + 1;     // not used

    //***********************************************************************
    //     Material parameters
    //***********************************************************************
    int media;
    media = MEDIACONSTANT; // number of different medias, ie 2: vacuum, metallicCylinder

    //***********************************************************************
    //     Wave excitation
    //***********************************************************************
    double rtau, tau, delay;
    rtau = 160.0e-12;
    tau = rtau / dt;
    delay = 3 * tau;

    double source[NUMBEROFITERATIONCONSTANT];
    InitialiseSource(nmax, source);
    WaveExcitation(tau, omega, delay, dt, source);

    //***********************************************************************
    //     Field arrays
    //***********************************************************************

    printf("\n\n");
    printf("     Allocating CPU memory for ex, ey and hz fields");
    printf("\n\n");

    double **ex, **ey, **hz; //fields in main grid

    ex = AllocateMemory(ie, jb, 0.0); //fields in main grid
    ey = AllocateMemory(ib, je, 0.0);
    hz = AllocateMemory(ie, je, 0.0);

    //
    //---- Front PML region
    //

    printf("\n\n");
    printf("     Allocating CPU memory for exbcf, eybcf and hzxbcf and hzybcf fields (front PML region)");
    printf("\n\n");

    double **exbcf; //fields in front PML region
    double **eybcf;
    double **hzxbcf;
    double **hzybcf;

    exbcf  = AllocateMemory(iefbc, jebc, 0.0); //fields in front PML region
    eybcf  = AllocateMemory(ibfbc, jebc, 0.0);
    hzxbcf = AllocateMemory(iefbc, jebc, 0.0);
    hzybcf = AllocateMemory(iefbc, jebc, 0.0);

    //
    //---- Back PML region
    //

    printf("\n\n");
    printf("     Allocating CPU memory for exbcb, eybcb and hzxbcb and hzybcb fields (back PML region)");
    printf("\n\n");

    double **exbcb; //fields in back PML region
    double **eybcb;
    double **hzxbcb;
    double **hzybcb;

    exbcb  = AllocateMemory(iefbc, jbbc, 0.0); //fields in back PML region
    eybcb  = AllocateMemory(ibfbc, jebc, 0.0);
    hzxbcb = AllocateMemory(iefbc, jebc, 0.0);
    hzybcb = AllocateMemory(iefbc, jebc, 0.0);

    //
    //---- Left PML region
    //

    printf("\n\n");
    printf("     Allocating CPU memory for exbcl, eybcl and hzxbcl and hzybcl fields (left PML region)");
    printf("\n\n");

    double **exbcl; //fields in left PML region
    double **eybcl;
    double **hzxbcl;
    double **hzybcl;

    exbcl  = AllocateMemory(iebc, jb, 0.0); //fields in left PML region
    eybcl  = AllocateMemory(iebc, je, 0.0);
    hzxbcl = AllocateMemory(iebc, je, 0.0);
    hzybcl = AllocateMemory(iebc, je, 0.0);

    //
    //---- Right PML region
    //
    //

    printf("\n\n");
    printf("     Allocating CPU memory for exbcr, eybcr and hzxbcr and hzybcr fields (right PML region)");
    printf("\n\n");

    double **exbcr; //fields in right PML region
    double **eybcr;
    double **hzxbcr;
    double **hzybcr;

    exbcr  = AllocateMemory(iebc, jb, 0.0); //fields in right PML region
    eybcr  = AllocateMemory(ibbc, je, 0.0);
    hzxbcr = AllocateMemory(iebc, je, 0.0);
    hzybcr = AllocateMemory(iebc, je, 0.0);

    //

    printf("\n\n");
    printf("     CPU RAM has been allocated for all field components");
    printf("\n\n");

    printCPUCount();

    //***********************************************************************
    //     Updating coefficients
    //***********************************************************************
    double eps[MEDIACONSTANT] = {1.0, 1.0}; // index=0 is for vacuum, index=1 is for the metallic cylinder
    double sig[MEDIACONSTANT] = {0.0, 1.0e+7};
    double mur[MEDIACONSTANT] = {1.0, 1.0};
    double sim[MEDIACONSTANT] = {0.0, 0.0};
    double ca[MEDIACONSTANT], cb[MEDIACONSTANT];
    double da[MEDIACONSTANT], db[MEDIACONSTANT];
    double eaf, haf;
    UpdateCoefficients(media, dt, dx, eaf, haf, epsz, muz, sig, eps, ca, cb, da, db, sim, mur);

    //***********************************************************************
    //     Geometry specification (main grid)
    //***********************************************************************
    //Initialize entire main grid to free space
    double **caex; // main grid coefficents
    double **cbex;
    caex = AllocateMemory(ie, jb, ca[0]);
    cbex = AllocateMemory(ie, jb, cb[0]);

    double **caey;
    double **cbey;
    caey = AllocateMemory(ib, je, ca[0]);
    cbey = AllocateMemory(ib, je, cb[0]);

    double **dahz;
    double **dbhz;
    dahz = AllocateMemory(ie, je, da[0]);
    dbhz = AllocateMemory(ie, je, db[0]);

    //Add metal cylinder
    double diam, rad, dist2;
    double temporaryi, temporaryj;
    int  icenter, jcenter;
    diam = 20;              // diameter of cylinder: 6 cm
    rad = diam / 2.0;       // radius of cylinder: 3 cm
    icenter = (4 * ie) / 5; // i-coordinate of cylinder's center
    jcenter = je / 2;       // j-coordinate of cylinder's center
    AddMetalCylinder(icenter, jcenter, ie, je, rad, temporaryi, temporaryj, dist2, ca, cb, caex, cbex, caey, cbey);

    //***********************************************************************
    //     Fill the PML regions
    //***********************************************************************
    double **caexbcf, **cbexbcf, **caexbcb, **cbexbcb; // pml coefficients
    caexbcf = AllocateMemory(iefbc, jebc, 0.0);
    cbexbcf = AllocateMemory(iefbc, jebc, 0.0);
    caexbcb = AllocateMemory(iefbc, jbbc, 0.0);
    cbexbcb = AllocateMemory(iefbc, jbbc, 0.0);

    double **caexbcl, **cbexbcl, **caexbcr, **cbexbcr;
    caexbcl = AllocateMemory(iebc, jb, 0.0);
    cbexbcl = AllocateMemory(iebc, jb, 0.0);
    caexbcr = AllocateMemory(iebc, jb, 0.0);
    cbexbcr = AllocateMemory(iebc, jb, 0.0);

    double **caeybcf, **cbeybcf, **caeybcb, **cbeybcb;
    caeybcf = AllocateMemory(ibfbc, jebc, 0.0);
    cbeybcf = AllocateMemory(ibfbc, jebc, 0.0);
    caeybcb = AllocateMemory(ibfbc, jebc, 0.0);
    cbeybcb = AllocateMemory(ibfbc, jebc, 0.0);

    double **caeybcl, **cbeybcl, **caeybcr, **cbeybcr;
    caeybcl = AllocateMemory(iebc, je, 0.0);
    cbeybcl = AllocateMemory(iebc, je, 0.0);
    caeybcr = AllocateMemory(ibbc, je, 0.0);
    cbeybcr = AllocateMemory(ibbc, je, 0.0);

    double **dahzxbcf, **dbhzxbcf, **dahzxbcb, **dbhzxbcb;
    dahzxbcf = AllocateMemory(iefbc, jebc, 0.0);
    dbhzxbcf = AllocateMemory(iefbc, jebc, 0.0);
    dahzxbcb = AllocateMemory(iefbc, jebc, 0.0);
    dbhzxbcb = AllocateMemory(iefbc, jebc, 0.0);

    double **dahzxbcl, **dbhzxbcl, **dahzxbcr, **dbhzxbcr;
    dahzxbcl = AllocateMemory(iebc, je, 0.0);
    dbhzxbcl = AllocateMemory(iebc, je, 0.0);
    dahzxbcr = AllocateMemory(iebc, je, 0.0);
    dbhzxbcr = AllocateMemory(iebc, je, 0.0);

    double **dahzybcf, **dbhzybcf, **dahzybcb, **dbhzybcb;
    dahzybcf = AllocateMemory(iefbc, jebc, 0.0);
    dbhzybcf = AllocateMemory(iefbc, jebc, 0.0);
    dahzybcb = AllocateMemory(iefbc, jebc, 0.0);
    dbhzybcb = AllocateMemory(iefbc, jebc, 0.0);

    double **dahzybcl, **dbhzybcl, **dahzybcr, **dbhzybcr;
    dahzybcl = AllocateMemory(iebc, je, 0.0);
    dbhzybcl = AllocateMemory(iebc, je, 0.0);
    dahzybcr = AllocateMemory(iebc, je, 0.0);
    dbhzybcr = AllocateMemory(iebc, je, 0.0);

    double delbc;
    delbc = (double)iebc * dx; // width of PML region (in mm)

    // SigmaMaximum, using polynomial grading (Nikolova part 4, p.30), rmax=reflectionMax in percent
    double sigmam;
    sigmam = -log(rmax / 100.0) * epsz * cc * (orderbc + 1) / (2 * delbc);

    // bcfactor comes from the polynomial grading equation: sigma_x = sigmaxMaximum * (x/d)^m, where d=width of PML, m=gradingOrder, (Nikolova part4, p.28)
    //  IMPORTANT: The conductivity (sigma) must use the "average" value at each mesh point as follows:
    //  sigma_x = sigma_Maximum/dx * Integral_from_x0_to_x1 of (x/d)^m dx,  where x0=currentx-0.5, x1=currentx+0.5   (Nikolova part 4, p.32)
    //  integrating gives: sigma_x = (sigmaMaximum / (dx * d^m * m+1)) * ( x1^(m+1) - x0^(m+1) )     (Nikolova part 4, p.32)
    //  the first part is "bcfactor", so, sigma_x = bcfactor * ( x1^(m+1) - x0^(m+1) )   (Nikolova part 4, p.32)
    // note: it's not exactly clear what the term eps[0] is for. It's probably to cover the case in which eps[0] is not equal to one (ie the main grid area next to the pml boundary is not vacuum)
    double bcfactor;
    bcfactor = eps[0] * sigmam / (dx * (pow(delbc, orderbc)) * (orderbc + 1));

    // FRONT region
    double ca1, cb1, da1, db1, sigmay, sigmays, y1, y2;
    FrontRegion(ie, iebc, iefbc, ibfbc, jebc, cb1, da1, db1, muz, epsz, ca1, orderbc, bcfactor, sigmay, sigmays, dt, dx, y1, y2, ca, cb, da, db, eps, cbex, caex, cbexbcf, caexbcf, caexbcl,
                cbexbcl, caexbcr, cbexbcr, caeybcf, cbeybcf, dahzybcf, dbhzybcf, dahzxbcf, dbhzxbcf);
    // BACK region
    BackRegion(ie, je, ibfbc, iebc, iefbc, jebc, bcfactor, ca1, cb1, da1, db1, dt, dx, epsz, muz, orderbc, sigmay, sigmays, y1, y2, ca, cb, da, db, eps, caex, cbex, caexbcb, cbexbcb, caeybcb,
               cbeybcb, caexbcl, cbexbcl, caexbcr, cbexbcr, dahzybcb, dbhzybcb, dahzxbcb, dbhzxbcb);
    // LEFT region
    double x1, x2, sigmax, sigmaxs;
    LeftRegion(je, iebc, jebc, bcfactor, ca1, cb1, da1, db1, dt, dx, epsz, muz, orderbc, sigmax, sigmaxs, x1, x2, ca, cb, da, db, eps, caey, cbey, caexbcl, cbexbcl, caeybcl, cbeybcl, caeybcb,
               cbeybcb, caeybcf, cbeybcf, dahzxbcb, dbhzxbcb, dahzybcf, dbhzybcf, dahzxbcf, dbhzxbcf, dahzxbcl, dbhzxbcl, dahzybcl, dbhzybcl);
    // RIGHT region
    LeftRegion(je, iebc, jebc, bcfactor, ca1, cb1, da1, db1, dt, dx, epsz, muz, orderbc, sigmax, sigmaxs, x1, x2, ca, cb, da, db, eps, caey, cbey, caexbcl, cbexbcl, caeybcl, cbeybcl, caeybcb,
               cbeybcb, caeybcf, cbeybcf, dahzxbcb, dbhzxbcb, dahzybcf, dbhzybcf, dahzxbcf, dbhzxbcf, dahzxbcl, dbhzxbcl, dahzybcl, dbhzybcl);

    //***********************************************************************
    //     Print variables (diagnostic)
    //***********************************************************************
    #if 0
            for (i = 0; i < nmax; i++) {
                printf("%d: source: %16.10g\n",i,source[i]);
            } /* iForLoop */

            // print main grid geometry
            printf("main grid:\n");
            for (i = 0; i < ib; i++) {
                printf("%3d: ",i);
                for (j = 0; j < jb; j++) {
                    ch = '.';
                    if (i < ie) {
                        if (caex[i][j] != ca[0]) {
                            ch = 'x';
                        } /* if */
                    } /* if */
                    if (j < je) {
                        if (caey[i][j] != ca[0]) {
                            if (ch == '.') {
                                ch = 'y';
                            } /* if */
                            else {
                                ch = 'B';
                            } /* else */
                        } /* if */
                    } /* if */
                    if ((i == is) && (j == js)) {
                        ch = 'S';
                    } /* if */
                    printf("%c",ch);
                } /* jForLoop */
                printf("\n");
            } /* iForLoop */
            printf("\n");
            // ---------------------------------------

            for (j = jebc-1; j >= 0; j--) {
                printf("back j:%3d: ",j);
                for (i = 0; i < ibfbc; i++) {
                    printf("[i:%3d: ",i);
    #if 0
                    if (i < iefbc) {
                        printf("ex:%+9.6g ",caexbcb[i][j]);
                    } /* if */
                    else {
                        printf("ex:------- ");
                    } /* else */
    #endif
    #if 0
                    printf("ey:%+9.6g ",caeybcb[i][j]);
    #endif
    #if 0
                    if (i < iefbc) {
                        printf("hzx:%+9.6g ",dahzxbcb[i][j]);
                    } /* if */
                    else {
                        printf("hzx:------- ");
                    } /* else */
    #endif
    #if 1
                    if (i < iefbc) {
                        printf("hzy:%+9.6g ",dahzybcb[i][j]);
                    } /* if */
                    else {
                        printf("hzy:------- ");
                    } /* else */
    #endif
                    printf("] ");
                } /* iForLoop */
                printf("\n");
            } /* jForLoop */


            // ----------------------main------------------------------
            for (j = je; j >= 0; j--) {
                printf("main j:%3d: ",j);
                for (i = 0; i < iebc; i++) {              // left
                    printf("[i:%3d: ",i);
    #if 0
                    printf("ex:%+9.6g ",caexbcl[i][j]);
    #endif
    #if 0
                    if (j < je) {
                        printf("ey:%+9.6g ",caeybcl[i][j]);
                    } /* if */
                    else {
                        printf("ey:--------- ");
                    } /* else */
    #endif
    #if 0
                    if (j < je) {
                        printf("hzx:%+9.6g ",dahzxbcl[i][j]);
                    } /* if */
                    else {
                        printf("hzx:--------- ");
                    } /* else */
    #endif
    #if 1
                    if  (j < je) {
                        printf("hzy:%+9.6g ",dahzybcl[i][j]);
                    } /* if */
                    else {
                        printf("hzy:--------- ");
                    } /* else */
    #endif
                    printf("] ");
                } /* iForLoop */


                for (i = 0; i < ib; i++) {            // center
                    printf("[i:%3d: ",i);
    #if 0
                    if (i < ie) {
                        printf("ex:%+9.6g ",caex[i][j]);
                    } /* if */
                    else {
                        printf("ex:--------- ");
                    } /* else */
    #endif
    #if 0
                    if (j < je) {
                        printf("ey:%+9.6g ",caey[i][j]);
                    } /* if */
                    else {
                        printf("ey:--------- ");
                    } /* else */
    #endif
    #if 1
                    if ((i < ie) && (j < je) ) {
                        printf("hz:%+9.6g  ",dahz[i][j]);
                    } /* if */
                    else {
                        printf("hz:---------- ");
                    } /* else */
    #endif


                    printf("] ");
                } /* iForLoop */


                for (i = 0; i < ibbc; i++) {     // right
                    printf("[i:%3d: ",i);
    #if 0
                    if (i < iebc) {
                        printf("ex:%+9.6g ",caexbcr[i][j]);
                    } /* if */
                    else {
                        printf("ex:---------");
                    } /* else */
    #endif
    #if 0
                    if (j < je) {
                        printf("ey:%+9.6g ",caeybcr[i][j]);
                    } /* if */
                    else {
                        printf("ey:--------- ");
                    } /* else */
    #endif
    #if 0
                    if ((j < je) && (i < iebc)) {
                        printf("hzx:%+9.6g ",dahzxbcr[i][j]);
                    } /* if */
                    else {
                        printf("hzx:--------- ");
                    } /* else */
    #endif
    #if 1
                    if ((j < je) && (i < iebc)) {
                        printf("hzy:%+9.6g ",dahzybcl[i][j]);
                    } /* if */
                    else {
                        printf("hzy:--------- ");
                    } /* else */
    #endif
                    printf("] ");
                } /* iForLoop */
                printf("\n");
            } /* jForLoop */


            // front -------------------------------------------------------
            for (j = jebc-1; j >= 0; j--) {
                printf("frnt j:%3d: ",j);
                for (i = 0; i < ibfbc; i++) {
                    printf("[i:%3d: ",i);
    #if 0
                    if (i < iefbc) {
                        printf("ex:%+9.6g ",caexbcf[i][j]);
                    } /* if */
                    else {
                        printf("ex:--------- ");
                    } /* else */
    #endif
    #if 0
                    printf("ey:%+9.6g ",caeybcf[i][j]);
    #endif
    #if 0
                    if (i < iefbc) {
                        printf("hzx:%+9.6g ",dahzxbcf[i][j]);
                    } /* if */
                    else {
                        printf("hzx:--------- ");
                    } /* else */
    #endif
    #if 1
                    if (i < iefbc) {
                        printf("hzy:%+9.6g ",dahzybcf[i][j]);
                    } /* if */
                    else {
                        printf("hzy:--------- ");
                    } /* else */
    #endif
                    printf("] ");
                } /* iForLoop */
                printf("\n");
            } /* jForLoop */

    #endif

    // all done with initialization, so can now start the simulation...
    //***********************************************************************
    //     BEGIN TIME-STEPPING LOOP
    //***********************************************************************
    // The plan is to allow for saving the output results by a passed seved name in the planned GUI.
    // Output Variables
    char outputFolder[] = "Output";
    char filename[BIGLINESIZE];
    double scaleValue;
    TimeSteppingLoop(centery, ie, iebc, is, je, iefbc, jebc, js, nmax, plottingInterval, ib, jb,
                     minimumValue, maximumValue, scaleValue,
                     source, filename, outputFolder,
                     caex, cbex, caey, cbey, caexbcb, cbexbcb, caexbcf, cbexbcf,
                     caexbcl, cbexbcl, caexbcr, cbexbcr, caeybcb, cbeybcb, caeybcf, cbeybcf,
                     caeybcl, cbeybcl, caeybcr, cbeybcr, dahz, dbhz, dahzxbcb, dbhzxbcb,
                     dahzybcb, dbhzybcb, dahzxbcf, dahzybcf, dbhzybcf, dbhzxbcf, dahzxbcl,
                     dbhzxbcl, dahzybcl, dbhzybcl, dahzybcr, dbhzybcr, dahzxbcr, dbhzxbcr,
                     ex, exbcf, exbcl, exbcr, exbcb, ey, eybcb, eybcf, eybcl,
                     eybcr, hz, hzxbcb, hzybcb, hzxbcf, hzybcf, hzxbcl, hzybcl,
                     hzxbcr, hzybcr,
                     runningMode,
                     debugMode
    );

    FreeMemoryUsages(
        ie, ib, iefbc, ibfbc, iebc, ibbc,
        eybcf, exbcf, hzxbcf, hzybcf,
        exbcb, eybcb, hzxbcb, hzybcb,
        exbcl, eybcl, hzxbcl, hzybcl,
        exbcr, eybcr, hzxbcr, hzybcr,
        caexbcf, cbexbcf, caexbcb, cbexbcb,
        caexbcl, cbexbcl, caexbcr, cbexbcr,
        caeybcf, cbeybcf, caeybcb, cbeybcb,
        caeybcl, cbeybcl, caeybcr, cbeybcr,
        dahzxbcf, dbhzxbcf, dahzxbcb, dbhzxbcb,
        dahzxbcl, dbhzxbcl, dahzxbcr, dbhzxbcr,
        dahzybcf, dbhzybcf, dahzybcb, dbhzybcb,
        dahzybcl, dbhzybcl, dahzybcr, dbhzybcr
    );
}

// standard C memory allocation for 2-D array
double **AllocateMemory(int imax, int jmax, double initialValue)
{
    int i, j;
    double **pointer;

    pointer = (double **)malloc(imax * sizeof(double *));

    //

    int nbytes = imax * sizeof(double *); 

    addCPUCount(nbytes,"AllocateMemory (double **):");

    //

    for (i = 0; i < imax; i++)
    {
        pointer[i] = (double *)malloc(jmax * sizeof(double));

        //

        int mbytes = jmax * sizeof(double); 

        addCPUCount(mbytes,"AllocateMemory (double *)");

        //


        for (j = 0; j < jmax; j++)
        {
            pointer[i][j] = initialValue;
        } /* jForLoop */
    }     /* iForLoop */
    return (pointer);
}

void
FreeMemory(int imax, double **pointer)
{
    int i;
    for (i = 0; i < imax; i++)
    {
        // printf("Loop: %d\n", i);
        free(pointer[i]);
    }
    // printf("End Loop\n");
    free(pointer);
}

void
InitialiseSource(int nmax, double source[])
{
    int i;
    for (i = 0; i < nmax; i++)
    {
        source[i] = 0.0;
    } /* iForLoop */
}

static void
WaveExcitation(double tau, double omega, double delay, double dt, double source[])
{
    int n;
    double temporary;
    for (n = 0; n < (int)(7.0 * tau); n++)
    {
        temporary = (double)n - delay;
        source[n] = sin(omega * (temporary)*dt) * exp(-((temporary * temporary) / (tau * tau)));
    }
}

static void
UpdateCoefficients(int media, double dt, double dx, double eaf, double haf, double epsz, double muz, double *sig, double *eps, double *ca, double *cb,
                   double *da, double *db, double *sim, double *mur)
{
    int i;
    for (i = 0; i < media; i++)
    {
        eaf = dt * sig[i] / (2.0 * epsz * eps[i]);     // Taflove1995 p.67
        ca[i] = (1.0 - eaf) / (1.0 + eaf);             // ditto
        cb[i] = dt / epsz / eps[i] / dx / (1.0 + eaf); // ditto

        haf = dt * sim[i] / (2.0 * muz * mur[i]);     // ditto
        da[i] = (1.0 - haf) / (1.0 + haf);            // ditto
        db[i] = dt / muz / mur[i] / dx / (1.0 + haf); // ditto
    }                                                 /* iForLoop */
}

static void
AddMetalCylinder(int icenter, int jcenter, int ie, int je, double rad, double temporaryi, double temporaryj, double dist2, double *ca, double *cb, double **caex,
                 double **cbex, double **caey, double **cbey)
{
    int i, j;
    for (i = 0; i < ie; i++)
    {
        for (j = 0; j < je; j++)
        {
            temporaryi = (double)(i - icenter);
            temporaryj = (double)(j - jcenter);
            dist2 = (temporaryi + 0.5) * (temporaryi + 0.5) + (temporaryj) * (temporaryj);
            if (dist2 <= (rad * rad))
            {
                caex[i][j] = ca[1];
                cbex[i][j] = cb[1];
            } /* if */
            // This looks tricky! Why can't caey/cbey use the same 'if' statement as caex/cbex above ??
            dist2 = (temporaryj + 0.5) * (temporaryj + 0.5) + (temporaryi) * (temporaryi);
            if (dist2 <= (rad * rad))
            {
                caey[i][j] = ca[1];
                cbey[i][j] = cb[1];
            } /* if */
        }     /* jForLoop */
    }         /* iForLoop */
}

static void
FrontRegion(int ie, int iebc, int iefbc, int ibfbc, int jebc, double cb1, double db1, double da1, double muz, double epsz, double ca1, double orderbc,
            double bcfactor, double sigmay, double sigmays, double dt, double dx, double y1, double y2, double ca[], double cb[], double da[], double db[],
            double eps[], double **cbex, double **caex, double **cbexbcf, double **caexbcf, double **caexbcl, double **cbexbcl, double **caexbcr,
            double **cbexbcr, double **caeybcf, double **cbeybcf, double **dahzybcf, double **dbhzybcf, double **dahzxbcf, double **dbhzxbcf)
{
    FrontRegionFirstLoop(iefbc, caexbcf, cbexbcf);
    FrontRegionSecondLoop(iefbc, jebc, bcfactor, ca1, cb1, dt, dx, epsz, orderbc, sigmay, y1, y2, eps, caexbcf, cbexbcf);
    FrontRegionThirdLoop(ie, bcfactor, ca1, cb1, dt, dx, epsz, orderbc, sigmay, eps, caex, cbex);
    FrontRegionFourthLoop(ca1, cb1, iebc, caex, cbex, caexbcl, cbexbcl, caexbcr, cbexbcr);
    FrontRegionFifthLoop(ibfbc, iefbc, jebc, bcfactor, da1, db1, dt, dx, epsz, muz, orderbc, sigmay, sigmays, y1, y2, ca, cb, da, db, eps, caeybcf, cbeybcf, dahzybcf, dbhzybcf, dahzxbcf, dbhzxbcf);
}

static void
FrontRegionFirstLoop(int iefbc, double **caexbcf, double **cbexbcf)
{
    int i;
    for (i = 0; i < iefbc; i++)
    { // IS THIS EVER USED? -- coef for the pec ex at j=0 set so that ex_t+1 = ex_t, but ex is never evaluated at j=0.
        caexbcf[i][0] = 1.0;
        cbexbcf[i][0] = 0.0;
    } /* iForLoop */
}

static void
FrontRegionSecondLoop(int iefbc, int jebc, double bcfactor, double ca1, double cb1, double dt, double dx, double epsz, double orderbc, double sigmay, double y1,
                      double y2, double eps[], double **caexbcf, double **cbexbcf)
{
    int i, j;
    for (j = 1; j < jebc; j++)
    {
        // calculate the coefs for the PML layer (except for the boundary at the main grid, which is a special case (see below))
        // LOCAL
        y1 = ((double)(jebc - j) + 0.5) * dx;                                  // upper bounds for point j      (re-adujsted for C indexes!)
        y2 = ((double)(jebc - j) - 0.5) * dx;                                  // lower bounds for point j
        sigmay = bcfactor * (pow(y1, (orderbc + 1)) - pow(y2, (orderbc + 1))); //   polynomial grading
        ca1 = exp(-sigmay * dt / (epsz * eps[0]));                             // exponential time step, Taflove1995 p.77,78
        cb1 = (1.0 - ca1) / (sigmay * dx);                                     // ditto, but note sign change from Taflove1995
        for (i = 0; i < iefbc; i++)
        {
            // Values from first loop, this should be more efficient?
            // caexbcf[i][0] = 1.0;
            // cbexbcf[i][0] = 0.0;
            // Values from second loop
            caexbcf[i][j] = ca1;
            cbexbcf[i][j] = cb1;
        } /* iForLoop */
    }     /* jForLoop */
}

static void
FrontRegionThirdLoop(int ie, double bcfactor, double ca1, double cb1, double dt, double dx, double epsz, double orderbc, double sigmay, double eps[],
                     double **caex, double **cbex)
{
    int i;
    // Probably can just use local version of Sigmay, ca1 and cb1 here.
    // LOCAL
    sigmay = bcfactor * pow((0.5 * dx), (orderbc + 1)); // calculate for the front edge of the pml at j=0 in the main grid  (half vacuum (sigma=0) and half pml)
    ca1 = exp(-sigmay * dt / (epsz * eps[0]));
    cb1 = (1 - ca1) / (sigmay * dx);
    for (i = 0; i < ie; i++)
    {
        caex[i][0] = ca1;
        cbex[i][0] = cb1;
    } /* iForLoop */
}

static void
FrontRegionFourthLoop(int iebc, double ca1, double cb1, double **caex, double **cbex, double **caexbcl, double **cbexbcl, double **caexbcr, double **cbexbcr)
{
    int i;
    // Fourth Loop
    for (i = 0; i < iebc; i++)
    { // this continues the front edge into the left and right grids
        caexbcl[i][0] = ca1;
        cbexbcl[i][0] = cb1;
        caexbcr[i][0] = ca1;
        cbexbcr[i][0] = cb1;
    } /* iForLoop */
}

static void
FrontRegionFifthLoop(int ibfbc, int iefbc, int jebc, double bcfactor, double da1, double db1, double dt, double dx, double epsz, double muz, double orderbc,
                     double sigmay, double sigmays, double y1, double y2, double ca[], double cb[], double da[], double db[], double eps[], double **caeybcf,
                     double **cbeybcf, double **dahzybcf, double **dbhzybcf, double **dahzxbcf, double **dbhzxbcf)
{
    int i, j;
    // Fifth Loop
    for (j = 0; j < jebc; j++)
    {                                                                          // for ey and hz  (which are offset spacially 1/2 dx from ex)
        y1 = ((double)(jebc - j) + 0.0) * dx;                                  // upper bounds for point j
        y2 = ((double)(jebc - j) - 1.0) * dx;                                  // lower bounds for point j
        sigmay = bcfactor * (pow(y1, (orderbc + 1)) - pow(y2, (orderbc + 1))); //   polynomial grading
        sigmays = sigmay * (muz / (epsz * eps[0]));                            // Taflove1995 p.182  (for no reflection: sigmaM = sigmaE * mu0/eps0)
        da1 = exp(-sigmays * dt / muz);                                        // exponential time step, Taflove1995 p.77,78
        db1 = (1.0 - da1) / (sigmays * dx);
        for (i = 0; i < iefbc; i++)
        {
            dahzybcf[i][j] = da1;
            dbhzybcf[i][j] = db1;
            dahzxbcf[i][j] = da[0]; // important note: hzx is Perpendicular to the front pml and so is not attenuated (sigma=0) (looks like vacuum)
            dbhzxbcf[i][j] = db[0]; // ditto
        }                           /* iForLoop */
        for (i = 0; i < ibfbc; i++)
        {
            caeybcf[i][j] = ca[0]; // important note: ey is Perpendicular to the front pml and so is not attenuated (sigma=0) (looks like vacuum)
            cbeybcf[i][j] = cb[0]; // ditto
        }                          /* iForLoop */
    }                              /* jForLoop */
}

static void
BackRegion(int ie, int je, int ibfbc, int iebc, int iefbc, int jebc, double bcfactor, double ca1, double cb1, double da1, double db1, double dt, double dx,
           double epsz, double muz, double orderbc, double sigmay, double sigmays, double y1, double y2, double ca[], double cb[], double da[], double db[],
           double eps[], double **caex, double **cbex, double **caexbcb, double **cbexbcb, double **caeybcb, double **cbeybcb, double **caexbcl,
           double **cbexbcl, double **caexbcr, double **cbexbcr, double **dahzybcb, double **dbhzybcb, double **dahzxbcb, double **dbhzxbcb)
{
    BackRegionFirstLoop(jebc, iefbc, caexbcb, cbexbcb);
    BackRegionSecondLoop(iefbc, jebc, bcfactor, ca1, cb1, dt, dx, epsz, orderbc, sigmay, y1, y2, eps, caexbcb, cbexbcb);
    sigmay = bcfactor * pow((0.5 * dx), (orderbc + 1)); // calculate for the front edge of the pml at j=0 in the main grid  (half vacuum (sigma=0) and half pml)
    ca1 = exp(-sigmay * dt / (epsz * eps[0]));
    cb1 = (1 - ca1) / (sigmay * dx);
    BackRegionThirdLoop(ie, je, bcfactor, ca1, cb1, dt, dx, epsz, orderbc, sigmay, eps, caex, cbex);
    BackRegionFourthLoop(je, ca1, cb1, iebc, caexbcl, cbexbcl, caexbcr, cbexbcr);
    BackRegionFifthLoop(ibfbc, iefbc, jebc, bcfactor, da1, db1, dt, dx, epsz, muz, orderbc, sigmay, sigmays, y1, y2, ca, cb, da, db, eps, caeybcb, cbeybcb, dahzybcb, dbhzybcb, dahzxbcb, dbhzxbcb);
}

static void
BackRegionFirstLoop(int jebc, int iefbc, double **caexbcb, double **cbexbcb)
{
    int i, j;
    for (j = jebc, i = 0; i < iefbc; i++)
    { // IS THIS EVER USED? -- coef for the pec ex at j=jebc set to ex_t+1 = ex_t
        caexbcb[i][jebc] = 1.0;
        cbexbcb[i][jebc] = 0.0;
    } /* iForLoop */
}

static void
BackRegionSecondLoop(int iefbc, int jebc, double bcfactor, double ca1, double cb1, double dt, double dx, double epsz, double orderbc, double sigmay, double y1,
                     double y2, double eps[], double **caexbcb, double **cbexbcb)
{
    int i, j;
    for (j = 1; j < jebc; j++)
    {                                                                          // calculate the coefs for the PML layer (except for the boundary at the main grid, which is a special case (see below))
        y1 = ((double)(j) + 0.5) * dx;                                         // upper bounds for point j         (re-adujsted for C indexes!)
        y2 = ((double)(j)-0.5) * dx;                                           // lower bounds for point j
        sigmay = bcfactor * (pow(y1, (orderbc + 1)) - pow(y2, (orderbc + 1))); //   polynomial grading
        ca1 = exp(-sigmay * dt / (epsz * eps[0]));                             // exponential time step
        cb1 = (1.0 - ca1) / (sigmay * dx);                                     // ditto, but note sign change from Taflove
        for (i = 0; i < iefbc; i++)
        {
            // Values from first loop, this should be more efficient?
            // caexbcb[i][jebc] = 1.0;
            // cbexbcb[i][jebc] = 0.0;
            // Values from second loop
            caexbcb[i][j] = ca1;
            cbexbcb[i][j] = cb1;
        } /* iForLoop */
    }     /* jForLoop */
}

static void
BackRegionThirdLoop(int ie, int je, double bcfactor, double ca1, double cb1, double dt, double dx, double epsz, double orderbc, double sigmay, double eps[],
                    double **caex, double **cbex)
{
    int i, j;
    for (i = 0; i < ie; i++)
    {
        caex[i][je] = ca1;
        cbex[i][je] = cb1;
    } /* iForLoop */
}

static void
BackRegionFourthLoop(int je, int iebc, double ca1, double cb1, double **caexbcl, double **cbexbcl, double **caexbcr, double **cbexbcr)
{
    int i, j;
    for (i = 0; i < iebc; i++)
    { // this continues the back edge into the left and right grids
        caexbcl[i][je] = ca1;
        cbexbcl[i][je] = cb1;
        caexbcr[i][je] = ca1;
        cbexbcr[i][je] = cb1;
    } /* iForLoop */
}

static void
BackRegionFifthLoop(int ibfbc, int iefbc, int jebc, double bcfactor, double da1, double db1, double dt, double dx, double epsz, double muz, double orderbc,
                    double sigmay, double sigmays, double y1, double y2, double ca[], double cb[], double da[], double db[], double eps[], double **caeybcb,
                    double **cbeybcb, double **dahzybcb, double **dbhzybcb, double **dahzxbcb, double **dbhzxbcb)
{
    int i, j;
    for (j = 0; j < jebc; j++)
    {                                                                          // for ey and hz  (which are offset spacially 1/2 dx from ex)
        y1 = ((double)(j) + 1.0) * dx;                                         // upper bounds for point j
        y2 = ((double)(j) + 0.0) * dx;                                         // lower bounds for point j
        sigmay = bcfactor * (pow(y1, (orderbc + 1)) - pow(y2, (orderbc + 1))); //   polynomial grading
        sigmays = sigmay * (muz / (epsz * eps[0]));
        da1 = exp(-sigmays * dt / muz);
        db1 = (1.0 - da1) / (sigmays * dx);
        for (i = 0; i < iefbc; i++)
        {
            dahzybcb[i][j] = da1;
            dbhzybcb[i][j] = db1;
            dahzxbcb[i][j] = da[0]; // important note: hzx is Perpendicular to the back pml and so is not attenuated (sigma=0) (looks like vacuum)
            dbhzxbcb[i][j] = db[0]; // ditto
        }                           /* iForLoop */
        for (i = 0; i < ibfbc; i++)
        {
            caeybcb[i][j] = ca[0]; // important note: ey is Perpendicular to the back pml and so is not attenuated (sigma=0) (looks like vacuum)
            cbeybcb[i][j] = cb[0]; // ditto
        }                          /* iForLoop */
    }                              /* jForLoop */
}

static void
LeftRegion(int je, int iebc, int jebc, double bcfactor, double ca1, double cb1, double da1, double db1, double dt, double dx, double epsz, double muz,
           double orderbc, double sigmax, double sigmaxs, double x1, double x2, double ca[], double cb[], double da[], double db[], double eps[], double **caey,
           double **cbey, double **caexbcl, double **cbexbcl, double **caeybcl, double **cbeybcl, double **caeybcb, double **cbeybcb, double **caeybcf,
           double **cbeybcf, double **dahzxbcb, double **dbhzxbcb, double **dahzybcf, double **dbhzybcf, double **dahzxbcf, double **dbhzxbcf, double **dahzxbcl,
           double **dbhzxbcl, double **dahzybcl, double **dbhzybcl)
{
    LeftRegionFirstLoop(je, caeybcl, cbeybcl);
    LeftRegionSecondLoop(je, iebc, jebc, bcfactor, ca1, cb1, epsz, x1, x2, dt, dx, orderbc, sigmax, eps, caeybcl, cbeybcl, caeybcf, cbeybcf, caeybcb, cbeybcb);
    LeftRegionThirdLoop(je, iebc, jebc, ca1, cb1, caey, cbey, caeybcf, cbeybcf, caeybcb, cbeybcb);
    LeftRegionFourthLoop(je, iebc, jebc, bcfactor, da1, db1, dt, dx, epsz, muz, orderbc, sigmax, sigmaxs, x1, x2, ca, cb, da, db, eps, caexbcl, cbexbcl, dahzybcl, dbhzybcl, dahzxbcl, dbhzxbcl, dahzxbcf, dbhzxbcf, dahzxbcb, dbhzxbcb);
}

static void
LeftRegionFirstLoop(int je, double **caeybcl, double **cbeybcl)
{
    int i, j;
    for (j = 0; j < je; j++)
    { // IS THIS EVER USED? -- coef for the pec ey at i=0 set to ey_t+1 = ey_t
        caeybcl[0][j] = 1.0;
        cbeybcl[0][j] = 0.0;
    } /* jForLoop */
}

static void
LeftRegionSecondLoop(int je, int iebc, int jebc, double bcfactor, double ca1, double cb1, double epsz, double x1, double x2, double dt, double dx,
                     double orderbc, double sigmax, double eps[], double **caeybcl, double **cbeybcl, double **caeybcf, double **cbeybcf, double **caeybcb,
                     double **cbeybcb)
{
    int i, j;
    for (i = 1; i < iebc; i++)
    {                                         // calculate the coefs for the PML layer (except for the boundary at the main grid, which is a special case (see below))
        x1 = ((double)(iebc - i) + 0.5) * dx; // upper bounds for point i    (re-adujsted for C indexes!)
        x2 = ((double)(iebc - i) - 0.5) * dx; // lower bounds for point i
        sigmax = bcfactor * (pow(x1, (orderbc + 1)) - pow(x2, (orderbc + 1)));
        ca1 = exp(-sigmax * dt / (epsz * eps[0]));
        cb1 = (1.0 - ca1) / (sigmax * dx);
        for (j = 0; j < je; j++)
        {
            caeybcl[i][j] = ca1;
            cbeybcl[i][j] = cb1;
        } /* jForLoop */
        for (j = 0; j < jebc; j++)
        { // fill in the front left and back left corners for ey
            caeybcf[i][j] = ca1;
            cbeybcf[i][j] = cb1;
            caeybcb[i][j] = ca1;
            cbeybcb[i][j] = cb1;
        } /* jForLoop */
    }     /* iForLoop */
}

static void
LeftRegionThirdLoop(int je, int iebc, int jebc, double ca1, double cb1, double **caey, double **cbey, double **caeybcf, double **cbeybcf, double **caeybcb,
                    double **cbeybcb)
{
    int i, j;
    for (j = 0; j < je; j++)
    {
        caey[0][j] = ca1;
        cbey[0][j] = cb1;
    } /* jForLoop */
    for (i = iebc, j = 0; j < jebc; j++)
    { // continue the left edge into the front and back grids
        caeybcf[iebc][j] = ca1;
        cbeybcf[iebc][j] = cb1;
        caeybcb[iebc][j] = ca1;
        cbeybcb[iebc][j] = cb1;
    } /* jForLoop */
}

static void
LeftRegionFourthLoop(int je, int iebc, int jebc, double bcfactor, double da1, double db1, double dt, double dx, double epsz, double muz, double orderbc,
                     double sigmax, double sigmaxs, double x1, double x2, double ca[], double cb[], double da[], double db[], double eps[], double **caexbcl,
                     double **cbexbcl, double **dahzybcl, double **dbhzybcl, double **dahzxbcl, double **dbhzxbcl, double **dahzxbcf, double **dbhzxbcf,
                     double **dahzxbcb, double **dbhzxbcb)
{
    int i, j;
    for (i = 0; i < iebc; i++)
    {                                         // for ex and hz  (which are offset spacially 1/2 dx from ey)
        x1 = ((double)(iebc - i) + 0.0) * dx; // upper bounds for point i    (re-adujsted for C indexes!)
        x2 = ((double)(iebc - i) - 1.0) * dx; // lower bounds for point i
        sigmax = bcfactor * (pow(x1, (orderbc + 1)) - pow(x2, (orderbc + 1)));
        sigmaxs = sigmax * (muz / (epsz * eps[0]));
        da1 = exp(-sigmaxs * dt / muz);
        db1 = (1 - da1) / (sigmaxs * dx);
        for (j = 0; j < je; j++)
        {
            dahzxbcl[i][j] = da1;
            dbhzxbcl[i][j] = db1;
            dahzybcl[i][j] = da[0]; // important note: hzy is Perpendicular to the left pml and so is not attenuated (sigma=0) (looks like vacuum)
            dbhzybcl[i][j] = db[0];
        } /* jForLoop */
        for (j = 0; j < jebc; j++)
        { // fill in the front left and back left corners for hzx
            dahzxbcf[i][j] = da1;
            dbhzxbcf[i][j] = db1;
            dahzxbcb[i][j] = da1;
            dbhzxbcb[i][j] = db1;
        } /* jForLoop */
        for (j = 1; j < je; j++)
        { // important note: ex is Perpendicular to the left pml and so is not attenuated (sigma=0) (looks like vacuum)
            caexbcl[i][j] = ca[0];
            cbexbcl[i][j] = cb[0];
        } /* jForLoop */
    }     /* iForLoop */
}

static void
RightRegion(int ie, int je, int iebc, double bcfactor, double ca1, double cb1, double epsz, double da1, double db1, double dt, double dx, int jebc, double muz,
            double orderbc, double sigmax, double sigmaxs, double x1, double x2, double ca[], double cb[], double da[], double db[], double eps[], double **caey,
            double **cbey, double **caeybcr, double **cbeybcr, double **caeybcf, double **cbeybcf, double **caeybcb, double **cbeybcb, double **caexbcr,
            double **cbexbcr, double **dahzxbcf, double **dbhzxbcf, double **dahzxbcb, double **dbhzxbcb, double **dahzxbcr, double **dbhzxbcr,
            double **dahzybcr, double **dbhzybcr)
{
    RightRegionFirstLoop(je, iebc, caeybcr, cbeybcr);
    RightRegionSecondLoop(ie, je, iebc, jebc, bcfactor, ca1, cb1, dt, dx, epsz, orderbc, sigmax, x1, x2, eps, caeybcr, cbeybcr, caeybcf, cbeybcf, caeybcb, cbeybcb);
    RightRegionThirdLoop(ie, je, iebc, jebc, bcfactor, ca1, cb1, dt, dx, epsz, orderbc, sigmax, eps, caey, cbey, caeybcb, cbeybcb, caeybcf, cbeybcf);
    RightRegionFourthLoop(ie, je, iebc, jebc, bcfactor, da1, db1, dt, dx, epsz, muz, orderbc, sigmax, sigmaxs, x1, x2, ca, cb, da, db, eps, caexbcr, cbexbcr, dahzxbcf, dbhzxbcf, dahzxbcb, dbhzxbcb, dahzxbcr, dbhzxbcr, dahzybcr, dbhzybcr);
}

static void
RightRegionFirstLoop(int je, int iebc, double **caeybcr, double **cbeybcr)
{
    int i, j;
    for (i = iebc, j = 0; j < je; j++)
    { // IS THIS EVER USED? -- coef for the pec ey at i=iebc set to ey_t+1 = ey_t
        caeybcr[i][j] = 1.0;
        cbeybcr[i][j] = 0.0;
    } /* jForLoop */
}

static void
RightRegionSecondLoop(int ie, int je, int iebc, int jebc, double bcfactor, double ca1, double cb1, double dt, double dx, double epsz, double orderbc,
                      double sigmax, double x1, double x2, double eps[], double **caeybcr, double **cbeybcr, double **caeybcf, double **cbeybcf, double **caeybcb,
                      double **cbeybcb)
{
    int i, j;
    for (i = 1; i < iebc; i++)
    {                                  // calculate the coefs for the PML layer (except for the boundary at the main grid, which is a special case (see below))
        x1 = ((double)(i) + 0.5) * dx; // upper bounds for point i        (re-adujsted for C indexes!)
        x2 = ((double)(i)-0.5) * dx;   // lower bounds for point i
        sigmax = bcfactor * (pow(x1, (orderbc + 1)) - pow(x2, (orderbc + 1)));
        ca1 = exp(-sigmax * dt / (epsz * eps[0]));
        cb1 = (1.0 - ca1) / (sigmax * dx);
        for (j = 0; j < je; j++)
        {
            caeybcr[i][j] = ca1;
            cbeybcr[i][j] = cb1;
        } /* jForLoop */
        for (j = 0; j < jebc; j++)
        { // fill in the front right and back right corners for ey
            caeybcf[i + iebc + ie][j] = ca1;
            cbeybcf[i + iebc + ie][j] = cb1;
            caeybcb[i + iebc + ie][j] = ca1;
            cbeybcb[i + iebc + ie][j] = cb1;
        } /* jForLoop */
    }     /* iForLoop */
}

static void
RightRegionThirdLoop(int ie, int je, int iebc, int jebc, double bcfactor, double ca1, double cb1, double dt, double dx, double epsz, double orderbc,
                     double sigmax, double eps[], double **caey, double **cbey, double **caeybcb, double **cbeybcb, double **caeybcf, double **cbeybcf)
{
    int i, j;
    sigmax = bcfactor * pow((0.5 * dx), (orderbc + 1)); // calculate for the right edge of the pml at x=ic in the main grid  (half vacuum (sigma=0) and half pml)
    ca1 = exp(-sigmax * dt / (epsz * eps[0]));
    cb1 = (1.0 - ca1) / (sigmax * dx);
    for (i = ie, j = 0; j < je; j++)
    {
        caey[i][j] = ca1;
        cbey[i][j] = cb1;
    } /* jForLoop */
    for (i = iebc + ie, j = 0; j < jebc; j++)
    { // continue the right edge into the front and back grids
        caeybcf[i][j] = ca1;
        cbeybcf[i][j] = cb1;
        caeybcb[i][j] = ca1;
        cbeybcb[i][j] = cb1;
    } /* jForLoop */
}

static void
RightRegionFourthLoop(int ie, int je, int iebc, int jebc, double bcfactor, double da1, double db1, double dt, double dx, double epsz, double muz, double orderbc,
                      double sigmax, double sigmaxs, double x1, double x2, double ca[], double cb[], double da[], double db[], double eps[], double **caexbcr,
                      double **cbexbcr, double **dahzxbcf, double **dbhzxbcf, double **dahzxbcb, double **dbhzxbcb, double **dahzxbcr, double **dbhzxbcr,
                      double **dahzybcr, double **dbhzybcr)
{
    int i, j;
    for (i = 0; i < iebc; i++)
    {                                  // for ex and hz  (which are offset spacially 1/2 dx from ey)
        x1 = ((double)(i) + 1.0) * dx; // upper bounds for point i         (re-adujsted for C indexes!)
        x2 = ((double)(i) + 0.0) * dx; // lower bounds for point i
        sigmax = bcfactor * (pow(x1, (orderbc + 1)) - pow(x2, (orderbc + 1)));
        sigmaxs = sigmax * (muz / (epsz * eps[0]));
        da1 = exp(-sigmaxs * dt / muz);
        db1 = (1 - da1) / (sigmaxs * dx);
        for (j = 0; j < je; j++)
        {
            dahzxbcr[i][j] = da1;
            dbhzxbcr[i][j] = db1;
            dahzybcr[i][j] = da[0]; // important note: hzy is Perpendicular to the right pml and so is not attenuated (sigma=0) (looks like vacuum)
            dbhzybcr[i][j] = db[0];
        } /* jForLoop */
        for (j = 0; j < jebc; j++)
        { // fill in the front right and back right corners for hzx
            dahzxbcf[i + ie + iebc][j] = da1;
            dbhzxbcf[i + ie + iebc][j] = db1;
            dahzxbcb[i + ie + iebc][j] = da1;
            dbhzxbcb[i + ie + iebc][j] = db1;
        } /* jForLoop */
        for (j = 1; j < je; j++)
        { // important note: ex is Perpendicular to the right pml and so is not attenuated (sigma=0) (looks like vacuum)
            caexbcr[i][j] = ca[0];
            cbexbcr[i][j] = cb[0];
        } /* jForLoop */
    }     /* iForLoop */
}


const char *TimeSteppingLoopUpdate_OCL_source =
"                                                                                                                                                                       \n"
"__kernel void                                                                                                                                                          \n"
"TimeSteppingLoopUpdateEXEYMain_loop_one(                                                                                                                               \n"
"   __global double *cl_ex,                                                                                                                                             \n"
"   __global double *cl_caex,                                                                                                                                           \n"
"   __global double *cl_cbex,                                                                                                                                           \n"
"   __global double *cl_hz,                                                                                                                                             \n"
"   int je,                                                                                                                                                             \n"
"   int jb                                                                                                                                                              \n"
")                                                                                                                                                                      \n"
"{                                                                                                                                                                      \n"
"   int i, j, pos_je, pos_jb, i_jb, i_je;                                                                                                                               \n"
"   i = (int) get_global_id(0);                                                                                                                                         \n"
"   i_je = je * i;                                                                                                                                                      \n"
"   i_jb = jb * i;                                                                                                                                                      \n"
"   for (j = 1; j < je; j++)                                                                                                                                            \n"
"   {                                                                                                                                                                   \n"
"       pos_je = i_je + j;                                                                                                                                              \n"
"       pos_jb = i_jb + j;                                                                                                                                              \n"
"       cl_ex[pos_jb] = cl_caex[pos_jb] * cl_ex[pos_jb] + cl_cbex[pos_jb] * (cl_hz[pos_je] - cl_hz[pos_je - 1]);                                                        \n"
"   }                                                                                                                                                                   \n"
"}                                                                                                                                                                      \n"
"                                                                                                                                                                       \n"
"                                                                                                                                                                       \n"
"__kernel void                                                                                                                                                          \n"
"TimeSteppingLoopUpdateEXEYMain_loop_two(                                                                                                                               \n"
"   __global double *cl_ey,                                                                                                                                             \n"
"   __global double *cl_caey,                                                                                                                                           \n"
"   __global double *cl_cbey,                                                                                                                                           \n"
"   __global double *cl_hz,                                                                                                                                             \n"
"   int je,                                                                                                                                                             \n"
"   int ie                                                                                                                                                              \n"
")                                                                                                                                                                      \n"
"{                                                                                                                                                                      \n"
"   int i, j, pos;                                                                                                                                                      \n"
"   i = je * (get_global_id(0)+1);                                                                                                                                      \n"
"   for (j = 0; j < je; j++)                                                                                                                                            \n"
"   {                                                                                                                                                                   \n"
"       pos = i + j;                                                                                                                                                    \n"
"       cl_ey[pos] = cl_caey[pos] * cl_ey[pos] + cl_cbey[pos] * (cl_hz[pos - je] - cl_hz[pos]);                                                                         \n"
"   }                                                                                                                                                                   \n"
"}                                                                                                                                                                      \n"
"                                                                                                                                                                       \n"
"                                                                                                                                                                       \n"
"__kernel void                                                                                                                                                          \n"
"TimeSteppingLoopUpdateMagneticFieldHZ(                                                                                                                                 \n"
"   __global double *cl_hz,                                                                                                                                             \n"
"   __global double *cl_dahz,                                                                                                                                           \n"
"   __global double *cl_dbhz,                                                                                                                                           \n"
"   __global double *cl_ex,                                                                                                                                             \n"
"   __global double *cl_ey,                                                                                                                                             \n"
"   int je,                                                                                                                                                             \n"
"   int ie,                                                                                                                                                             \n"
"   int jb                                                                                                                                                              \n"
")                                                                                                                                                                      \n"
"{                                                                                                                                                                      \n"
"   int i, j, pos_je, pos_jb, i_jb, i_je;                                                                                                                               \n"
"   i = get_global_id(0);                                                                                                                                               \n"
"   i_je = je * i;                                                                                                                                                      \n"
"   i_jb = jb * i;                                                                                                                                                      \n"
"   for (j = 0; j < je; j++)                                                                                                                                            \n"
"   {                                                                                                                                                                   \n"
"       pos_je = i_je + j;                                                                                                                                              \n"
"       pos_jb = i_jb + j;                                                                                                                                              \n"
"       cl_hz[pos_je] = cl_dahz[pos_je] * cl_hz[pos_je] + cl_dbhz[pos_je] * (cl_ex[pos_jb + 1] - cl_ex[pos_jb] + cl_ey[pos_je] - cl_ey[pos_je + je]);                   \n"
"   }                                                                                                                                                                   \n"
"}                                                                                                                                                                      \n"
"                                                                                                                                                                       \n"
"                                                                                                                                                                       \n"
"__kernel void                                                                                                                                                          \n"
"TimeSteppingLoopPlotFields_OCL(                                                                                                                                        \n"
"   __global uchar *gif_frame,                                                                                                                                          \n"
"   __global double *cl_hz,                                                                                                                                             \n"
"   double minimumValue,                                                                                                                                                \n"
"   double scaleValue,                                                                                                                                                  \n"
"   int ie,                                                                                                                                                             \n"
"   int je                                                                                                                                                              \n"
")                                                                                                                                                                      \n"
"{                                                                                                                                                                      \n"
"   int i, j, j_hz, iValue, pos_hz, pos_gif;                                                                                                                            \n"
"   double temporary;                                                                                                                                                   \n"
"   j = (get_global_id(0));                                                                                                                                             \n"
"   j_hz = j*ie;                                                                                                                                                        \n"
"   for (i = 0; i < ie; i++)                                                                                                                                            \n"
"   {                                                                                                                                                                   \n"
"       pos_hz = (i*je) + j;                                                                                                                                            \n"
"       pos_gif = j_hz + i;                                                                                                                                             \n"
"       temporary = cl_hz[pos_hz];                                                                                                                                      \n"
"       temporary = (temporary - minimumValue) * scaleValue;                                                                                                            \n"
"       iValue = (int)(temporary);                                                                                                                                      \n"
"       if (iValue < 0)                                                                                                                                                 \n"
"       {                                                                                                                                                               \n"
"            iValue = 0;                                                                                                                                                \n"
"       }                                                                                                                                                               \n"
"       if (iValue > 255)                                                                                                                                               \n"
"       {                                                                                                                                                               \n"
"           iValue = 255;                                                                                                                                               \n"
"       }                                                                                                                                                               \n"
"       gif_frame[pos_hz] = iValue;                                                                                                                                    \n"
"   }                                                                                                                                                                   \n"
"}                                                                                                                                                                      \n"
"                                                                                                                                                                       \n"
"                                                                                                                                                                       \n";


static void
TimeSteppingLoop(
    int centery, int ie, int iebc, int is, int je, int iefbc, int jebc, int js, int nmax, int plottingInterval, int ib, int jb,
    double minimumValue, double maximumValue, double scaleValue,
    double source[], char filename[], char outputFolder[],
    double **caex, double **cbex, double **caey, double **cbey, double **caexbcb, double **cbexbcb, double **caexbcf, double **cbexbcf,
    double **caexbcl, double **cbexbcl, double **caexbcr, double **cbexbcr, double **caeybcb, double **cbeybcb, double **caeybcf, double **cbeybcf,
    double **caeybcl, double **cbeybcl, double **caeybcr, double **cbeybcr, double **dahz, double **dbhz, double **dahzxbcb, double **dbhzxbcb,
    double **dahzybcb, double **dbhzybcb, double **dahzxbcf, double **dahzybcf, double **dbhzybcf, double **dbhzxbcf, double **dahzxbcl,
    double **dbhzxbcl, double **dahzybcl, double **dbhzybcl, double **dahzybcr, double **dbhzybcr, double **dahzxbcr, double **dbhzxbcr,
    double **ex, double **exbcf, double **exbcl, double **exbcr, double **exbcb, double **ey, double **eybcb, double **eybcf, double **eybcl,
    double **eybcr, double **hz, double **hzxbcb, double **hzybcb, double **hzxbcf, double **hzybcf, double **hzxbcl, double **hzybcl,
    double **hzxbcr, double **hzybcr,
    enum RunningMode runningMode,
    int debugMode
)
{
    int i, j, n, gif_width, gif_height, palette_size;
    gif_width = ie;
    gif_height = je;
    palette_size = 768;
    FILE *file_hz, *file_ex, *file_ey;

    //

    printf("\n");
    printf("     Updates will be performed using: ");

    switch (runningMode)
      {
       default:
       case Standard:
                     printf(" CPU sequentially \n\n");
                     break;

       case OpenMP:
                     printf(" CPU using OpenMP \n\n");
                     break;

       case OpenCL:
                     printf(" OpenCL on the NVidia GPU \n\n");
                     break;
      }

    //

    if (debugMode == 1)
    {
        char Standard_Loc[] = "Output/Standard", OpenMP_Loc[] = "Output/OpenMP", OpenCL_Loc[] = "Output/OpenCL";
        char *FileName, *FileName_HZ, *FileName_EX, *FileName_EY;

        struct stat st = {0};
        if (stat(outputFolder, &st) == -1)
        {
            mkdir(outputFolder, 0700);
        }
        
        if (stat(Standard_Loc, &st) == -1)
        {
            mkdir(Standard_Loc, 0700);
        }
        
        if (stat(OpenMP_Loc, &st) == -1)
        {
            mkdir(OpenMP_Loc, 0700);
        }
        
        if (stat(OpenCL_Loc, &st) == -1)
        {
            mkdir(OpenCL_Loc, 0700);
        }

        switch (runningMode)
        {
            default:
            case Standard:
                FileName = buildFilename(Standard_Loc, ie, je);
                break;

            case OpenMP:
                FileName = buildFilename(OpenMP_Loc, ie, je);
                break;

            case OpenCL:
                FileName = buildFilename(OpenCL_Loc, ie, je);
                break;
        }

        FileName_HZ = (char *) malloc(strlen(FileName)+strlen("_hz.txt"));

    //

    int zbytes = strlen(FileName)+strlen("_hz.txt");

    addCPUCount(zbytes,"AllocateMemory (char *):");

    //


        FileName_EX = (char *) malloc(strlen(FileName)+strlen("_ex.txt"));

    //

    int xbytes = strlen(FileName)+strlen("_ex.txt");

    addCPUCount(xbytes,"AllocateMemory (char *):");

    //


        FileName_EY = (char *) malloc(strlen(FileName)+strlen("_ey.txt"));

    //

    int ybytes = strlen(FileName)+strlen("_ey.txt");

    addCPUCount(ybytes,"AllocateMemory (char *):");

    //


        memcpy(FileName_HZ, FileName, strlen(FileName));
        memcpy(FileName_EX, FileName, strlen(FileName));
        memcpy(FileName_EY, FileName, strlen(FileName));

        // HZ
        file_hz = fopen(strcat(FileName_HZ, "_hz.txt"), "w");
        if (file_hz == NULL)
        {
            printf("Error opening file\n");
            exit(1);
        }

        // EX
        file_ex = fopen(strcat(FileName_EX, "_ex.txt"), "w");
        if (file_ex == NULL)
        {
            printf("Error opening file!\n");
            exit(1);
        }

        // EY
        file_ey = fopen(strcat(FileName_EY, "_ey.txt"), "w");
        if (file_ey == NULL)
        {
            printf("Error opening file!\n");
            exit(1);
        }
    }

    uint8_t palette[palette_size], rgb;
    rgb = 0x00;

    for (i=0; i<(palette_size); i++)
    {
        palette[i] = rgb;
        if (i%3 == 0 && i != 0)
        {
            rgb++;
        }
    }

    /* create a GIF */
    //ge_GIF *hz_gif = ge_new_gif(
    //    "yee_hz.gif",              /* file name */
    //    gif_width, gif_height,  /* canvas size */
    //    palette,                /* palette */
    //    8,                      /* palette depth == log2(# of colors) */
    //    0                       /* infinite loop */
    //);

    // /* create a GIF */
    // ge_GIF *ey_gif = ge_new_gif(
    //     "yee_ey.gif",              /* file name */
    //     gif_width, gif_height,  /* canvas size */
    //     palette,                /* palette */
    //     8,                      /* palette depth == log2(# of colors) */
    //     0                       /* infinite loop */
    // );

    // /* create a GIF */
    // ge_GIF *ex_gif = ge_new_gif(
    //     "yee_ex.gif",              /* file name */
    //     gif_width, gif_height,  /* canvas size */
    //     palette,                /* palette */
    //     8,                      /* palette depth == log2(# of colors) */
    //     0                       /* infinite loop */
    // );

    // Transform 2D arrays to 1D
    double *single_ex, *single_ey, *single_hz, *single_caex, *single_cbex, *single_caey, *single_cbey, *single_dahz, *single_dbhz;
    int start, total;

    start = time(NULL);

    cl_platform_id platform;
    cl_uint platforms, devices;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_uint ret;
    cl_int  iret;

    cl_mem cl_gif_frame, cl_hz, cl_ex, cl_ey, cl_dahz, cl_dbhz, cl_caex, cl_cbex, cl_caey, cl_cbey;

    if (runningMode == OpenCL)
    {
        // 1. Get a platform.
        ret = clGetPlatformIDs( 1,
                                &platform,
                                NULL );

        // 2. Find a gpu device.
        ret =  clGetDeviceIDs(  platform,
                                CL_DEVICE_TYPE_GPU,
                                1,
                                &device,
                                NULL);

        // 3. Create a context and command queue on that device.
        context = clCreateContext(NULL,
                                  1,
                                  &device,
                                  NULL, NULL, &iret);

#if !defined(CL_VERSION_2_0) || defined(__NVCC__)

        printf("     Using NVCC version of OpenCL ");
        printf("\n\n");

        queue = clCreateCommandQueue(context,
                                     device,
                                     0,
                                     &iret);

#else

        queue = clCreateCommandQueueWithProperties(context,
                                                   device,
                                                   0,
                                                   &ret);
#endif

        // Minimal error check.
        if(queue == NULL) {
            printf("Compute device setup failed\n");
        }

        // 4. Perform runtime source compilation
        program = clCreateProgramWithSource(context,
                                            1,
                                            &TimeSteppingLoopUpdate_OCL_source,
                                            NULL, &iret );

        const char *options = " ";

        ret = clBuildProgram(program,
                            1,
                            &device,
                            options, NULL, NULL );

        // Print compiler error messages
        if(ret != CL_SUCCESS)
        {
            printf("clBuildProgram failed: %d\n", ret);
            char buf[0x10000];
            clGetProgramBuildInfo(program,
                                  device,
                                  CL_PROGRAM_BUILD_LOG,
                                  0x10000,
                                  buf,
                                  NULL);
            printf("\n%s\n", buf);
        }
        else
        {
            printf("     OpenCL program compiled correctly");
            printf("\n\n");
        }

        ret = clGetPlatformIDs(1, &platform, &platforms);

        ret = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, &devices);

        single_ex = TransformPointerToVector(ie, jb, ex);
        single_ey = TransformPointerToVector(ib, je, ey);
        single_hz = TransformPointerToVector(ie, je, hz);
        single_caex = TransformPointerToVector(ie, jb, caex);
        single_cbex = TransformPointerToVector(ie, jb, cbex);
        single_caey = TransformPointerToVector(ib, je, caey);
        single_cbey = TransformPointerToVector(ib, je, cbey);
        single_dahz = TransformPointerToVector(ie, je, dahz);
        single_dbhz = TransformPointerToVector(ie, je, dbhz);

        cl_ex = clCreateBuffer(
            context,
            CL_MEM_READ_WRITE,
            ib * je * sizeof(double),
            NULL, NULL
        );

        cl_ey = clCreateBuffer(
            context,
            CL_MEM_READ_WRITE,
            ib * je * sizeof(double),
            NULL, NULL
        );

        cl_hz = clCreateBuffer(
            context,
            CL_MEM_READ_WRITE,
            ie * je * sizeof(double),
            NULL, NULL
        );

        cl_dahz = clCreateBuffer(
            context,
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            ie * je * sizeof(double),
            &(single_dahz[0]), NULL
        );

        cl_dbhz = clCreateBuffer(
            context,
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            ie * je * sizeof(double),
            &(single_dbhz[0]), NULL
        );

        cl_gif_frame = clCreateBuffer(
            context,
            CL_MEM_READ_WRITE,
            (ie * je) * sizeof(uint8_t),
            NULL, NULL
        );

        cl_caex = clCreateBuffer(
            context,
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            ie * jb * sizeof(double),
            &(single_caex[0]), &iret
        );

        cl_cbex = clCreateBuffer(
            context,
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            ie * jb * sizeof(double),
            &(single_cbex[0]), &iret
        );

        cl_caey = clCreateBuffer(
            context,
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            ib * je * sizeof(double),
            &(single_caey[0]), &iret
        );

        cl_cbey = clCreateBuffer(
            context,
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            ib * je * sizeof(double),
            &(single_cbey[0]), &iret
        );
    }

    //
    //---- Iteration over timesteps 
    //

    fprintf(stdout,"\n"); 

    for (n = 0; n < nmax; n++)
    { 
        fprintf(stdout,"     Iteration %4d (n) of %4d (nmax) ", n, nmax);
        fprintf(stdout,"\n"); 

        //***********************************************************************
        //     Update electric fields (EX and EY) in main grid
        //***********************************************************************
        // Note the 4 edges, ey left (i=0), ey right (i=ie), ex bottom (j=0) and ex top (j=je) are evaluated in the pml section
        switch (runningMode)
        {
            default:
            case Standard:
                TimeSteppingLoopUpdateEXEYMain_STD(ie, je, caex, cbex, caey, cbey, ex, ey, hz);
                break;

            case OpenMP:
                TimeSteppingLoopUpdateEXEYMain_OMP(ie, je, caex, cbex, caey, cbey, ex, ey, hz);
                break;

            // Needs to be changed to OCL Source format
            case OpenCL:
                TimeSteppingLoopUpdateEXEYMain_OCL(ie, je, ib, jb, single_caex, single_cbex, single_caey, single_cbey, single_ex, single_ey, single_hz, context, program, queue, cl_hz, cl_ex, cl_ey, cl_caex, cl_cbex, cl_caey, cl_cbey);
                break;
        }

        //***********************************************************************
        // Update EX in PML regions
        //***********************************************************************
        switch (runningMode)
        {
            default:
            case Standard:
            case OpenMP:
                TimeSteppingLoopUpdateEXPML(ie, iebc, iefbc, je, jebc, caex, cbex, caexbcb, cbexbcb, caexbcf, cbexbcf, caexbcl, cbexbcl, caexbcr, cbexbcr, ex, exbcf, exbcl, exbcr, exbcb, hz, hzxbcb, hzybcb, hzxbcf, hzybcf, hzxbcl, hzybcl, hzxbcr, hzybcr);
                break;

            case OpenCL:
                TimeSteppingLoopUpdateEXPML_OCL(ie, iebc, iefbc, je, jebc, jb, single_caex, single_cbex, caexbcb, cbexbcb, caexbcf, cbexbcf, caexbcl, cbexbcl, caexbcr, cbexbcr, single_ex, exbcf, exbcl, exbcr, exbcb, single_hz, hzxbcb, hzybcb, hzxbcf, hzybcf, hzxbcl, hzybcl, hzxbcr, hzybcr);
                break;
        }

        //***********************************************************************
        // Update EY in PML regions
        //***********************************************************************
        switch (runningMode)
        {
            default:
            case Standard:
            case OpenMP:
                TimeSteppingLoopUpdateEYPML(ie, iebc, iefbc, je, jebc, caey, caeybcb, cbeybcb, caeybcf, cbeybcf, caeybcl, cbeybcl, caeybcr, cbeybcr, cbey, ey, eybcb, eybcf, eybcl, eybcr, hz, hzxbcb, hzybcb, hzxbcf, hzybcf, hzxbcl, hzybcl, hzxbcr, hzybcr);
                break;

            case OpenCL:
                TimeSteppingLoopUpdateEYPML_OCL(ie, iebc, iefbc, je, jebc, single_caey, caeybcb, cbeybcb, caeybcf, cbeybcf, caeybcl, cbeybcl, caeybcr, cbeybcr, single_cbey, single_ey, eybcb, eybcf, eybcl, eybcr, single_hz, hzxbcb, hzybcb, hzxbcf, hzybcf, hzxbcl, hzybcl, hzxbcr, hzybcr);
                break;
        }
        
        //***********************************************************************
        // Update magnetic fields (HZ) in main grid
        //***********************************************************************
        switch (runningMode)
        {
            default:
            case Standard:
                TimeSteppingLoopUpdateMagneticFieldHZ_STD(ie, is, je, js, n, source, dahz, dbhz, ex, ey, hz);
                break;

            case OpenMP:
                TimeSteppingLoopUpdateMagneticFieldHZ_OMP(ie, is, je, js, n, source, dahz, dbhz, ex, ey, hz);
                break;

            // Needs to be changed to OCL Source format
            case OpenCL:
                TimeSteppingLoopUpdateMagneticFieldHZ_OCL(ie, is, je, js, n, ib, jb, source, single_dahz, single_dbhz, single_ex, single_ey, single_hz, context, program, queue, cl_hz, cl_ex, cl_ey, cl_dahz, cl_dbhz);
                break;
        }
        
        //***********************************************************************
        // Update HZX in PML regions
        //***********************************************************************
        switch (runningMode)
        {
            default:
            case Standard:
            case OpenMP:
                TimeSteppingLoopUpdateHZXPML(ie, iebc, iefbc, je, jebc, dahzxbcb, dbhzxbcb, dahzxbcf, dbhzxbcf, dahzxbcl, dbhzxbcl, dahzxbcr, dbhzxbcr, ey, eybcb, eybcf, eybcl, eybcr, hzxbcb, hzxbcf, hzxbcl, hzxbcr);
                break;

            case OpenCL:
                TimeSteppingLoopUpdateHZXPML_OCL(ie, iebc, iefbc, je, jebc, dahzxbcb, dbhzxbcb, dahzxbcf, dbhzxbcf, dahzxbcl, dbhzxbcl, dahzxbcr, dbhzxbcr, single_ey, eybcb, eybcf, eybcl, eybcr, hzxbcb, hzxbcf, hzxbcl, hzxbcr);
                break;
        }

        //***********************************************************************
        // Update HZY in PML regions
        //***********************************************************************
        switch (runningMode)
        {
            default:
            case Standard:
            case OpenMP:
                TimeSteppingLoopUpdateHZYPML(ie, iebc, iefbc, je, jebc, dahzybcb, dbhzybcb, dahzxbcf, dbhzxbcf, dahzybcf, dbhzybcf, dahzybcl, dbhzybcl, dahzybcr, dbhzybcr, ex, exbcb, exbcf, exbcl, exbcr, hzybcb, hzybcf, hzybcl, hzybcr);
                break;

            case OpenCL:
                TimeSteppingLoopUpdateHZYPML_OCL(ie, iebc, iefbc, je, jebc, jb, dahzybcb, dbhzybcb, dahzxbcf, dbhzxbcf, dahzybcf, dbhzybcf, dahzybcl, dbhzybcl, dahzybcr, dbhzybcr, single_ex, exbcb, exbcf, exbcl, exbcr, hzybcb, hzybcf, hzybcl, hzybcr);
                break;
        }
        
        //***********************************************************************
        // Plot fields
        //***********************************************************************
        uint8_t *hz_gif_frame;
        switch (runningMode)
        {
            default:
            case Standard:
                hz_gif_frame = TimeSteppingLoopPlotFields_STD(centery, ie, je, n, plottingInterval, minimumValue, maximumValue, scaleValue, filename, outputFolder, ex, ey, hz);
                break;

            case OpenMP:
                hz_gif_frame = TimeSteppingLoopPlotFields_OMP(centery, ie, je, n, plottingInterval, minimumValue, maximumValue, scaleValue, filename, outputFolder, ex, ey, hz);
                break;

            // Needs to be changed to OCL Source format
            case OpenCL:
                hz_gif_frame = TimeSteppingLoopPlotFields_OCL(centery, ie, je, n, plottingInterval, minimumValue, maximumValue, scaleValue, filename, outputFolder, single_ex, single_ey, single_hz, context, program, queue, cl_gif_frame, cl_hz);
                break;
        }
        
        //hz_gif->frame = hz_gif_frame;
        //ge_add_frame(hz_gif, 1);

        // uint8_t *ey_gif_frame = TimeSteppingLoopPlotFields(centery, ie, je, n, plottingInterval, minimumValue, maximumValue, scaleValue, filename, outputFolder, ex, ey, hz);
        // ey_gif->frame = ey_gif_frame;
        // ge_add_frame(ey_gif, 1);

        // uint8_t *ex_gif_frame = TimeSteppingLoopPlotFields(centery, ie, je, n, plottingInterval, minimumValue, maximumValue, scaleValue, filename, outputFolder, ex, ey, hz);
        // ex_gif->frame = ex_gif_frame;
        // ge_add_frame(ex_gif, 1);

        if (debugMode == 1)
        {
            if (runningMode == OpenCL)
            {
                int max = ie*je;
                for (i = 0; i < ie; i++)  {
                    for (j = 0; j < je; j++) {
                        fprintf(file_hz, "%.17f\n", single_hz[(i*je)+j]);
                        fprintf(file_ex, "%.17f\n", single_ex[(i*jb)+j]);
                        fprintf(file_ey, "%.17f\n", single_ey[(i*je)+j]);
                    }
                }
            }
            else 
            {
                for (i = 0; i < ie; i++)  {
                    for (j = 0; j < je; j++) {
                        fprintf(file_hz, "%.17f\n", hz[i][j]);
                        fprintf(file_ex, "%.17f\n", ex[i][j]);
                        fprintf(file_ey, "%.17f\n", ey[i][j]);
                    }
                }
            }
        }
    }
    /* remember to close the GIF */
    // ge_close_gif(hz_gif);
    // ge_close_gif(ey_gif);
    // ge_close_gif(ex_gif);

    if (debugMode == 1)
    {
        fclose(file_hz);
        fclose(file_ex);
        fclose(file_ey);
    }

    if (runningMode == OpenCL){
        free(single_ex);
        free(single_ey);
        free(single_hz);
        free(single_caex);
        free(single_cbex);
        free(single_caey);
        free(single_cbey);
        free(single_dahz);
        free(single_dbhz);

        clReleaseMemObject(cl_hz);
        clReleaseMemObject(cl_ex);
        clReleaseMemObject(cl_ey);
        clReleaseMemObject(cl_gif_frame);
        clReleaseMemObject(cl_dahz);
        clReleaseMemObject(cl_dbhz);
        clReleaseMemObject(cl_caex);
        clReleaseMemObject(cl_cbex);
        clReleaseMemObject(cl_caey);
        clReleaseMemObject(cl_cbey);

        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
    }
    else {
        FreeMemory(ie, ex);
        FreeMemory(ib, ey);
        FreeMemory(ie, hz);
        FreeMemory(ie, caex);
        FreeMemory(ie, cbex);
        FreeMemory(ib, caey);
        FreeMemory(ib, cbey);
        FreeMemory(ie, dahz);
        FreeMemory(ie, dbhz);
    }

    int stop = time(NULL);
    total = stop - start;

    printf("\n\n");
    printf("     Time taken for FDTD Time Stepping Loop: %d seconds (X=%d,Y=%d)", total,ie,je);
    printf("\n\n");
}

static char*
buildFilename(char* src, int ie, int je)
{
    int i;
    char ie_string[32], je_string[32];

    sprintf(ie_string, "%d", ie);
    sprintf(je_string, "%d", je);
    strcat(src, "/");
    strcat(src, ie_string);
    strcat(src, "_");
    strcat(src, je_string);

    return src;
}

static void
TimeSteppingLoopUpdateEXPML(int ie, int iebc, int iefbc, int je, int jebc, double **caex, double **cbex, double **caexbcb, double **cbexbcb, double **caexbcf,
                            double **cbexbcf, double **caexbcl, double **cbexbcl, double **caexbcr, double **cbexbcr, double **ex, double **exbcf,
                            double **exbcl, double **exbcr, double **exbcb, double **hz, double **hzxbcb, double **hzybcb, double **hzxbcf, double **hzybcf,
                            double **hzxbcl, double **hzybcl, double **hzxbcr, double **hzybcr)
{
    // FRONT
    TimeSteppingLoopUpdateEXPMLFront(ie, iebc, iefbc, jebc, caex, cbex, caexbcf, cbexbcf, ex, exbcf, hz, hzxbcf, hzybcf);
    // BACK
    TimeSteppingLoopUpdateEXPMLBack(ie, iebc, je, iefbc, jebc, caex, cbex, caexbcb, cbexbcb, ex, exbcb, hz, hzxbcb, hzybcb);
    // LEFT
    TimeSteppingLoopUpdateEXPMLLeft(ie, iebc, iefbc, je, jebc, caexbcl, cbexbcl, exbcl, hzxbcb, hzybcb, hzxbcf, hzybcf, hzxbcl, hzybcl);
    // RIGHT
    TimeSteppingLoopUpdateEXPMLRight(ie, iebc, iefbc, je, jebc, caexbcr, cbexbcr, exbcr, hzxbcb, hzybcb, hzxbcf, hzybcf, hzxbcr, hzybcr);
}

static void
TimeSteppingLoopUpdateEXPMLFront(int ie, int iebc, int iefbc, int jebc, double **caex, double **cbex, double **caexbcf, double **cbexbcf, double **ex,
                                 double **exbcf, double **hz, double **hzxbcf, double **hzybcf)
{
    int i, j;
    for (i = 0; i < iefbc; i++)
    {
        for (j = 1; j < jebc; j++)
        { // don't Evaluate exbcf at j=0, as it is the PEC
            // note: sign change in the second term from main grid!! ... (due to the Exponential time stepping algorithm?)
            exbcf[i][j] = caexbcf[i][j] * exbcf[i][j] - cbexbcf[i][j] * (hzxbcf[i][j - 1] + hzybcf[i][j - 1] - hzxbcf[i][j] - hzybcf[i][j]);
        } /* jForLoop */
    }     /* iForLoop */

    for (j = 0, i = 0; i < ie; i++)
    { // fill in the edge for ex at j=0 main grid  (ties in the pml with the main grid)
        ex[i][j] = caex[i][j] * ex[i][j] - cbex[i][j] * (hzxbcf[iebc + i][jebc - 1] + hzybcf[iebc + i][jebc - 1] - hz[i][j]);
    } /* iForLoop */
}

static void
TimeSteppingLoopUpdateEXPMLBack(int ie, int iebc, int je, int iefbc, int jebc, double **caex, double **cbex, double **caexbcb, double **cbexbcb, double **ex,
                                double **exbcb, double **hz, double **hzxbcb, double **hzybcb)
{
    int i, j;
    // Locals
    double **l_exbcb, **l_ex;
    l_exbcb = exbcb;
    l_ex = ex;

    for (i = 0; i < iefbc; i++)
    {
        for (j = 1; j < jebc; j++)
        { // don't Evaluate exbcb at j=jebc, as it is the PEC, also dont eval at j=0 as this point is the same as j=je on the main grid
            l_exbcb[i][j] = caexbcb[i][j] * l_exbcb[i][j] - cbexbcb[i][j] * (hzxbcb[i][j - 1] + hzybcb[i][j - 1] - hzxbcb[i][j] - hzybcb[i][j]);
        } /* jForLoop */
    }     /* iForLoop */

    exbcb = l_exbcb;

    j = je;
    for (i = 0; i < ie; i++)
    { // fill in the edge for ex at j=je main grid  (ties in the pml with the main grid)
        l_ex[i][j] = caex[i][j] * l_ex[i][j] - cbex[i][j] * (hz[i][j - 1] - hzxbcb[iebc + i][0] - hzybcb[iebc + i][0]);
    } /* iForLoop */
    ex = l_ex;
}

static void
TimeSteppingLoopUpdateEXPMLLeft(int ie, int iebc, int iefbc, int je, int jebc, double **caexbcl, double **cbexbcl, double **exbcl, double **hzxbcb,
                                double **hzybcb, double **hzxbcf, double **hzybcf, double **hzxbcl, double **hzybcl)
{
    int i, j;
    for (i = 0; i < iebc; i++)
    {
        // don't Evaluate exbcl at j=0, j=0 is a special case, it needs data from the "front grid" see below
        // likewise, don't Evaluate exbcl at j=je, j=je is a special case, it needs data from the "back grid" see below
        for (j = 1; j < je; j++)
        {
            exbcl[i][j] = caexbcl[i][j] * exbcl[i][j] - cbexbcl[i][j] * (hzxbcl[i][j - 1] + hzybcl[i][j - 1] - hzxbcl[i][j] - hzybcl[i][j]);
        } /* jForLoop */
    }     /* iForLoop */

    for (j = 0, i = 0; i < iebc; i++)
    { // exbcl at j=0 case, uses data from the "front grid"
        exbcl[i][j] = caexbcl[i][j] * exbcl[i][j] - cbexbcl[i][j] * (hzxbcf[i][jebc - 1] + hzybcf[i][jebc - 1] - hzxbcl[i][j] - hzybcl[i][j]);
    } /* iForLoop */

    for (j = je, i = 0; i < iebc; i++)
    { // exbcl at j=je case, uses data from the "back grid"
        exbcl[i][j] = caexbcl[i][j] * exbcl[i][j] - cbexbcl[i][j] * (hzxbcl[i][j - 1] + hzybcl[i][j - 1] - hzxbcb[i][0] - hzybcb[i][0]);
    } /* iForLoop */
}

static void
TimeSteppingLoopUpdateEXPMLRight(int ie, int iebc, int iefbc, int je, int jebc, double **caexbcr, double **cbexbcr, double **exbcr, double **hzxbcb,
                                 double **hzybcb, double **hzxbcf, double **hzybcf, double **hzxbcr, double **hzybcr)
{
    int i, j;

    // Locals
    double **l_exbcr;
    l_exbcr = exbcr;

    for (i = 0; i < iebc; i++)
    {
        // don't Evaluate exbcr at j=0, j=0 is a special case, it needs data from the "front grid" see below
        // likewise, don't Evaluate exbcr at j=je, j=je is a special case, it needs data from the "back grid" see below
        for (j = 1; j < je; j++)
        {
            l_exbcr[i][j] = caexbcr[i][j] * l_exbcr[i][j] - cbexbcr[i][j] * (hzxbcr[i][j - 1] + hzybcr[i][j - 1] - hzxbcr[i][j] - hzybcr[i][j]);
        } /* jForLoop */
    }     /* iForLoop */

    j = 0;
    for (i = 0; i < iebc; i++)
    { // exbcr at j=0 case, uses data from the "front grid" (on the right side)
        l_exbcr[i][j] = caexbcr[i][j] * l_exbcr[i][j] - cbexbcr[i][j] * (hzxbcf[iebc + ie + i][jebc - 1] + hzybcf[iebc + ie + i][jebc - 1] - hzxbcr[i][j] - hzybcr[i][j]);
    } /* iForLoop */

    j = je;
    for (i = 0; i < iebc; i++)
    { // exbcr at j=je case, uses data from the "back grid"
        l_exbcr[i][j] = caexbcr[i][j] * l_exbcr[i][j] - cbexbcr[i][j] * (hzxbcr[i][j - 1] + hzybcr[i][j - 1] - hzxbcb[iebc + ie + i][0] - hzybcb[iebc + ie + i][0]);
    } /* iForLoop */
    exbcr = l_exbcr;
}

static void
TimeSteppingLoopUpdateEYPML(int ie, int iebc, int iefbc, int je, int jebc, double **caey, double **caeybcb, double **cbeybcb, double **caeybcf, double **cbeybcf,
                            double **caeybcl, double **cbeybcl, double **caeybcr, double **cbeybcr, double **cbey, double **ey, double **eybcb, double **eybcf,
                            double **eybcl, double **eybcr, double **hz, double **hzxbcb, double **hzybcb, double **hzxbcf, double **hzybcf, double **hzxbcl,
                            double **hzybcl, double **hzxbcr, double **hzybcr)
{
    // FRONT
    TimeSteppingLoopUpdateEYPMLFront(iefbc, jebc, caeybcf, cbeybcf, eybcf, hzxbcf, hzybcf);
    // BACK
    TimeSteppingLoopUpdateEYPMLBack(iefbc, jebc, caeybcb, cbeybcb, eybcb, hzxbcb, hzybcb);
    // LEFT
    TimeSteppingLoopUpdateEYPMLLeft(iebc, je, caey, caeybcl, cbeybcl, cbey, ey, eybcl, hz, hzxbcl, hzybcl);
    // RIGHT
    TimeSteppingLoopUpdateEYPMLRight(ie, iebc, je, caey, caeybcr, cbeybcr, cbey, ey, eybcr, hz, hzxbcr, hzybcr);
}

static void
TimeSteppingLoopUpdateEYPMLFront(int iefbc, int jebc, double **caeybcf, double **cbeybcf, double **eybcf, double **hzxbcf, double **hzybcf)
{
    int i, j;
    for (i = 1; i < iefbc; i++)
    { // don't Evaluate eybcf at i=0 or iefbc, as it is the PEC
        for (j = 0; j < jebc; j++)
        {
            // note: sign change in the second term from main grid!!
            eybcf[i][j] = caeybcf[i][j] * eybcf[i][j] - cbeybcf[i][j] * (hzxbcf[i][j] + hzybcf[i][j] - hzxbcf[i - 1][j] - hzybcf[i - 1][j]);
        } /* jForLoop */
    }     /* iForLoop */
}

static void
TimeSteppingLoopUpdateEYPMLBack(int iefbc, int jebc, double **caeybcb, double **cbeybcb, double **eybcb, double **hzxbcb, double **hzybcb)
{
    int i, j;
    // Locals

    for (i = 1; i < iefbc; i++)
    { // don't Evaluate eybcb at i=0 or iefbc, as it is the PEC
        for (j = 0; j < jebc; j++)
        {
            eybcb[i][j] = caeybcb[i][j] * eybcb[i][j] - cbeybcb[i][j] * (hzxbcb[i][j] + hzybcb[i][j] - hzxbcb[i - 1][j] - hzybcb[i - 1][j]);
        } /* jForLoop */
    }     /* iForLoop */
}

static void
TimeSteppingLoopUpdateEYPMLLeft(int iebc, int je, double **caey, double **caeybcl, double **cbeybcl, double **cbey, double **ey, double **eybcl, double **hz,
                                double **hzxbcl, double **hzybcl)
{
    int i, j;
    // Locals

    for (i = 1; i < iebc; i++)
    { // don't Evaluate eybcb at i=0, as it is the PEC
        for (j = 0; j < je; j++)
        {
            eybcl[i][j] = caeybcl[i][j] * eybcl[i][j] - cbeybcl[i][j] * (hzxbcl[i][j] + hzybcl[i][j] - hzxbcl[i - 1][j] - hzybcl[i - 1][j]);
        } /* jForLoop */
    }     /* iForLoop */

    i = 0;
    for (j = 0; j < je; j++)
    { // fill in the edge for ey at i=0 main grid  (ties in the pml with the main grid)
        ey[i][j] = caey[i][j] * ey[i][j] - cbey[i][j] * (hz[i][j] - hzxbcl[iebc - 1][j] - hzybcl[iebc - 1][j]);
    } /* jForLoop */
}

static void
TimeSteppingLoopUpdateEYPMLRight(int ie, int iebc, int je, double **caey, double **caeybcr, double **cbeybcr, double **cbey, double **ey, double **eybcr,
                                 double **hz, double **hzxbcr, double **hzybcr)
{
    int i, j;
    // Locals

    for (i = 1; i < iebc; i++)
    { // don't Evaluate eybcr at i=iebc, as it is the PEC, also dont eval at i=0 as this point is the same as i=ie on the main grid
        for (j = 0; j < je; j++)
        {
            eybcr[i][j] = caeybcr[i][j] * eybcr[i][j] - cbeybcr[i][j] * (hzxbcr[i][j] + hzybcr[i][j] - hzxbcr[i - 1][j] - hzybcr[i - 1][j]);
        } /* jForLoop */
    }     /* iForLoop */

    i = ie;
    for (j = 0; j < je; j++)
    { // fill in the edge for ey at i=ie main grid  (ties in the pml with the main grid)
        ey[i][j] = caey[i][j] * ey[i][j] - cbey[i][j] * (hzxbcr[0][j] + hzybcr[0][j] - hz[i - 1][j]);
    } /* jForLoop */
}

static void
TimeSteppingLoopUpdateHZXPML(int ie, int iebc, int iefbc, int je, int jebc, double **dahzxbcb, double **dbhzxbcb, double **dahzxbcf, double **dbhzxbcf,
                             double **dahzxbcl, double **dbhzxbcl, double **dahzxbcr, double **dbhzxbcr, double **ey, double **eybcb, double **eybcf,
                             double **eybcl, double **eybcr, double **hzxbcb, double **hzxbcf, double **hzxbcl, double **hzxbcr)
{
    // FRONT
    TimeSteppingLoopUpdateHZXPMLFront(iefbc, jebc, dahzxbcf, dbhzxbcf, eybcf, hzxbcf);
    // BACK
    TimeSteppingLoopUpdateHZXPMLBack(iefbc, jebc, dahzxbcb, dbhzxbcb, eybcb, hzxbcb);
    // LEFT
    TimeSteppingLoopUpdateHZXPMLLeft(je, iebc, dahzxbcl, dbhzxbcl, ey, eybcl, hzxbcl);
    // RIGHT
    TimeSteppingLoopUpdateHZXPMLRight(ie, je, iebc, dahzxbcr, dbhzxbcr, ey, eybcr, hzxbcr);
}

static void
TimeSteppingLoopUpdateHZXPMLFront(int iefbc, int jebc, double **dahzxbcf, double **dbhzxbcf, double **eybcf, double **hzxbcf)
{
    int i, j;
    for (i = 0; i < iefbc; i++)
    {
        for (j = 0; j < jebc; j++)
        {
            // note: sign change in the second term from main grid!!
            hzxbcf[i][j] = dahzxbcf[i][j] * hzxbcf[i][j] - dbhzxbcf[i][j] * (eybcf[i + 1][j] - eybcf[i][j]);
        } /* jForLoop */
    }     /* iForLoop */
}

static void
TimeSteppingLoopUpdateHZXPMLBack(int iefbc, int jebc, double **dahzxbcb, double **dbhzxbcb, double **eybcb, double **hzxbcb)
{
    int i, j;
    for (i = 0; i < iefbc; i++)
    {
        for (j = 0; j < jebc; j++)
        {
            hzxbcb[i][j] = dahzxbcb[i][j] * hzxbcb[i][j] - dbhzxbcb[i][j] * (eybcb[i + 1][j] - eybcb[i][j]);
        } /* jForLoop */
    }     /* iForLoop */
}

static void
TimeSteppingLoopUpdateHZXPMLLeft(int je, int iebc, double **dahzxbcl, double **dbhzxbcl, double **ey, double **eybcl, double **hzxbcl)
{
    int i, j;
    for (i = 0; i < (iebc - 1); i++)
    { // don't evaluate hzxbcl at i=iebc-1 as it needs ey from main grid, see below
        for (j = 0; j < je; j++)
        {
            hzxbcl[i][j] = dahzxbcl[i][j] * hzxbcl[i][j] - dbhzxbcl[i][j] * (eybcl[i + 1][j] - eybcl[i][j]);
        } /* jForLoop */
    }     /* iForLoop */

    for (i = (iebc - 1), j = 0; j < je; j++)
    { // fix-up hzxbcl at i=iebc-1
        hzxbcl[i][j] = dahzxbcl[i][j] * hzxbcl[i][j] - dbhzxbcl[i][j] * (ey[0][j] - eybcl[i][j]);
    } /* jForLoop */
}

static void
TimeSteppingLoopUpdateHZXPMLRight(int ie, int je, int iebc, double **dahzxbcr, double **dbhzxbcr, double **ey, double **eybcr, double **hzxbcr)
{
    int i, j;
    for (i = 1; i < iebc; i++)
    { // don't evaluate hzxbcl at i=0 as it needs ey from main grid, see below
        for (j = 0; j < je; j++)
        {
            hzxbcr[i][j] = dahzxbcr[i][j] * hzxbcr[i][j] - dbhzxbcr[i][j] * (eybcr[i + 1][j] - eybcr[i][j]);
        } /* jForLoop */
    }     /* iForLoop */

    for (i = 0, j = 0; j < je; j++)
    { // fix-up hzxbcl at i=0
        hzxbcr[i][j] = dahzxbcr[i][j] * hzxbcr[i][j] - dbhzxbcr[i][j] * (eybcr[i + 1][j] - ey[ie][j]);
    } /* jForLoop */
}

static void
TimeSteppingLoopUpdateHZYPML(int ie, int iebc, int iefbc, int je, int jebc, double **dahzybcb, double **dbhzybcb, double **dahzxbcf, double **dbhzxbcf,
                             double **dahzybcf, double **dbhzybcf, double **dahzybcl, double **dbhzybcl, double **dahzybcr, double **dbhzybcr, double **ex,
                             double **exbcb, double **exbcf, double **exbcl, double **exbcr, double **hzybcb, double **hzybcf, double **hzybcl, double **hzybcr)
{
    // FRONT
    TimeSteppingLoopUpdateHZYPMLFront(ie, iebc, iefbc, je, jebc, dahzxbcf, dbhzxbcf, dahzybcf, dbhzybcf, ex, exbcf, exbcl, exbcr, hzybcf);
    // BACK
    TimeSteppingLoopUpdateHZYPMLBack(ie, iebc, iefbc, je, jebc, dahzybcb, dbhzybcb, ex, exbcb, exbcl, exbcr, hzybcb);
    // LEFT
    TimeSteppingLoopUpdateHZYPMLLeft(ie, iebc, je, jebc, dahzybcl, dbhzybcl, exbcl, hzybcl);
    // RIGHT
    TimeSteppingLoopUpdateHZYPMLRight(iebc, je, dahzybcr, dbhzybcr, exbcr, hzybcr);
}

static void
TimeSteppingLoopUpdateHZYPMLFront(int ie, int iebc, int iefbc, int je, int jebc, double **dahzxbcf, double **dbhzxbcf, double **dahzybcf, double **dbhzybcf,
                                  double **ex, double **exbcf, double **exbcl, double **exbcr, double **hzybcf)
{
    int i, j;
    for (i = 0; i < iefbc; i++)
    {
        for (j = 0; j < (jebc - 1); j++)
        { // don't evaluate hzxbcf at j=jebc-1 as it needs data from main,left,right grids, see below
            // note: sign change in the second term from main grid!!
            hzybcf[i][j] = dahzybcf[i][j] * hzybcf[i][j] - dbhzybcf[i][j] * (exbcf[i][j] - exbcf[i][j + 1]);
        } /* jForLoop */
    }     /* iForLoop */

    for (j = (jebc - 1), i = 0; i < iebc; i++)
    { // fix-up hzybcf at j=jebc-1, with left grid
        hzybcf[i][j] = dahzybcf[i][j] * hzybcf[i][j] - dbhzybcf[i][j] * (exbcf[i][j] - exbcl[i][0]);
    } /* iForLoop */

    for (j = (jebc - 1), i = 0; i < ie; i++)
    { // fix-up hzybcf at j=jebc-1, with main grid
        hzybcf[iebc + i][j] = dahzybcf[iebc + i][j] * hzybcf[iebc + i][j] - dbhzybcf[iebc + i][j] * (exbcf[iebc + i][j] - ex[i][0]);
    } /* iForLoop */

    for (j = (jebc - 1), i = 0; i < iebc; i++)
    { // fix-up hzybcf at j=jebc-1, with right grid
        hzybcf[iebc + ie + i][j] = dahzybcf[iebc + ie + i][j] * hzybcf[iebc + ie + i][j] - dbhzybcf[iebc + ie + i][j] * (exbcf[iebc + ie + i][j] - exbcr[i][0]);
    } /* iForLoop */
}

static void
TimeSteppingLoopUpdateHZYPMLBack(int ie, int iebc, int iefbc, int je, int jebc, double **dahzybcb, double **dbhzybcb, double **ex, double **exbcb,
                                 double **exbcl, double **exbcr, double **hzybcb)
{
    int i, j;
    for (i = 0; i < iefbc; i++)
    {
        for (j = 1; j < jebc; j++)
        { // don't evaluate hzxbcb at j=0 as it needs data from main,left,right grids, see below
            hzybcb[i][j] = dahzybcb[i][j] * hzybcb[i][j] - dbhzybcb[i][j] * (exbcb[i][j] - exbcb[i][j + 1]);
        } /* jForLoop */
    }     /* iForLoop */

    for (j = 0, i = 0; i < iebc; i++)
    { // fix-up hzybcb at j=0, with left grid
        hzybcb[i][j] = dahzybcb[i][j] * hzybcb[i][j] - dbhzybcb[i][j] * (exbcl[i][je] - exbcb[i][j + 1]);
    } /* iForLoop */

    for (j = 0, i = 0; i < ie; i++)
    { // fix-up hzybcb at j=0, with main grid
        hzybcb[iebc + i][j] = dahzybcb[iebc + i][j] * hzybcb[iebc + i][j] - dbhzybcb[iebc + i][j] * (ex[i][je] - exbcb[iebc + i][j + 1]);
    } /* iForLoop */

    for (j = 0, i = 0; i < iebc; i++)
    { // fix-up hzybcb at j=0, with right grid
        hzybcb[iebc + ie + i][j] = dahzybcb[iebc + ie + i][j] * hzybcb[iebc + ie + i][j] - dbhzybcb[iebc + ie + i][j] * (exbcr[i][je] - exbcb[iebc + ie + i][j + 1]);
    } /* iForLoop */
}

static void
TimeSteppingLoopUpdateHZYPMLLeft(int ie, int iebc, int je, int jebc, double **dahzybcl, double **dbhzybcl, double **exbcl, double **hzybcl)
{
    int i, j;
    for (i = 0; i < iebc; i++)
    {
        for (j = 0; j < je; j++)
        {
            hzybcl[i][j] = dahzybcl[i][j] * hzybcl[i][j] - dbhzybcl[i][j] * (exbcl[i][j] - exbcl[i][j + 1]);
        } /* jForLoop */
    }     /* iForLoop */
}

static void
TimeSteppingLoopUpdateHZYPMLRight(int iebc, int je, double **dahzybcr, double **dbhzybcr, double **exbcr, double **hzybcr)
{
    int i, j;
    // Locals
    double **l_hzybcr;
    l_hzybcr = hzybcr;

    for (i = 0; i < iebc; i++)
    {
        for (j = 0; j < je; j++)
        {
            l_hzybcr[i][j] = dahzybcr[i][j] * l_hzybcr[i][j] - dbhzybcr[i][j] * (exbcr[i][j] - exbcr[i][j + 1]);
        } /* jForLoop */
    }     /* iForLoop */

    hzybcr = l_hzybcr;
}

// Base Methods

static void
TimeSteppingLoopUpdateEXEYMain_STD(int ie, int je, double **caex, double **cbex, double **caey, double **cbey, double **ex, double **ey, double **hz)
{
    int i, j;

    for (i = 0; i < ie; i++)
    {
        for (j = 1; j < je; j++)
        { // dont do ex at j=0 or j=je, it will be done in the PML section
            ex[i][j] = caex[i][j] * ex[i][j] + cbex[i][j] * (hz[i][j] - hz[i][j - 1]);
        } /* jForLoop */
    }     /* iForLoop */

    for (i = 1; i < ie; i++)
    { // dont do ey at i=0 or i=ie,  it will be done in the PML section
        for (j = 0; j < je; j++)
        {
            ey[i][j] = caey[i][j] * ey[i][j] + cbey[i][j] * (hz[i - 1][j] - hz[i][j]);
        } /* jForLoop */
    }     /* iForLoop */
}

static void
TimeSteppingLoopUpdateMagneticFieldHZ_STD(int ie, int is, int je, int js, int n, double source[], double **dahz, double **dbhz, double **ex, double **ey,
                                      double **hz)
{
    int i, j;
    for (i = 0; i < ie; i++)
    {
        for (j = 0; j < je; j++)
        {
            hz[i][j] = dahz[i][j] * hz[i][j] + dbhz[i][j] * (ex[i][j + 1] - ex[i][j] + ey[i][j] - ey[i + 1][j]);
        } /* jForLoop */
    }     /* iForLoop */

    hz[is][js] = source[n];
}

static uint8_t *TimeSteppingLoopPlotFields_STD(int centery, int ie, int je, int n, int plottingInterval, double minimumValue, double maximumValue, double scaleValue,
                                        char filename[], char outputFolder[], double **ex, double **ey, double **hz)
{
    int i, j;
    uint8_t *gif_frame = (uint8_t *) malloc(ie*je);

    //

    int tbytes = ie*je;

    addCPUCount(tbytes,"AllocateMemory (uint8_t *):");

    //


    if (plottingInterval == 0)
    {
        plottingInterval = 2;

        // Checks the existance of the
        struct stat st = {0};
        if (stat(outputFolder, &st) == -1)
        {
            mkdir(outputFolder, 0700);
        }

        int iValue;
        //int count;
        double minimumValue, maximumValue, temporary, **hzlocal;
        minimumValue = -0.1;
        maximumValue = 0.1;

        scaleValue = 256.0 / (maximumValue - minimumValue);
        hzlocal = hz;

        //count = 0;

        for (j = 0; j < je; j++)
        {
            for (i = 0; i < ie; i++)
            {
                temporary = hzlocal[i][j];
                temporary = (temporary - minimumValue) * scaleValue;
                iValue = (int)(temporary);
                if (iValue < 0)
                {
                    iValue = 0;
                } /* if */
                if (iValue > 255)
                {
                    iValue = 255;
                } /* if */

                int pos;
                pos = ((i) * je) + j;

                gif_frame[pos] = iValue;
            } /* xForLoop */
        } /* yForLoop */
    } /* if */
    plottingInterval--;
    
    return gif_frame;
}

// OpenMP

static void
TimeSteppingLoopUpdateEXEYMain_OMP(int ie, int je, double **caex, double **cbex, double **caey, double **cbey, double **ex, double **ey, double **hz)
{
    int i, j;

    // Locals
    double **l_ex, **l_ey;
    l_ex = ex;
    l_ey = ey;

    #pragma omp parallel for private(i, j) shared(l_ex)
    for (i = 0; i < ie; i++)
    {
        for (j = 1; j < je; j++)
        { // dont do ex at j=0 or j=je, it will be done in the PML section
            l_ex[i][j] = caex[i][j] * l_ex[i][j] + cbex[i][j] * (hz[i][j] - hz[i][j - 1]);
        } /* jForLoop */
    }     /* iForLoop */

    ex = l_ex;

    #pragma omp parallel for private(i, j) shared(l_ey)
    for (i = 1; i < ie; i++)
    { // dont do ey at i=0 or i=ie,  it will be done in the PML section
        for (j = 0; j < je; j++)
        {
            l_ey[i][j] = caey[i][j] * l_ey[i][j] + cbey[i][j] * (hz[i - 1][j] - hz[i][j]);
        } /* jForLoop */
    }     /* iForLoop */

    ey = l_ey;
}

static void
TimeSteppingLoopUpdateMagneticFieldHZ_OMP(int ie, int is, int je, int js, int n, double source[], double **dahz, double **dbhz, double **ex, double **ey,
                                      double **hz)
{
    int i, j;

    // Locals
    double **l_hz;
    l_hz = hz;

    #pragma omp parallel for private(i, j) shared(l_hz)
    for (i = 0; i < ie; i++)
    {
        for (j = 0; j < je; j++)
        {
            l_hz[i][j] = dahz[i][j] * l_hz[i][j] + dbhz[i][j] * (ex[i][j + 1] - ex[i][j] + ey[i][j] - ey[i + 1][j]);
        } /* jForLoop */
    }     /* iForLoop */
    hz = l_hz;

    hz[is][js] = source[n];
}

static uint8_t *TimeSteppingLoopPlotFields_OMP(int centery, int ie, int je, int n, int plottingInterval, double minimumValue, double maximumValue, double scaleValue,
                                        char filename[], char outputFolder[], double **ex, double **ey, double **hz)
{
    int i, j;
    uint8_t *gif_frame = (uint8_t *) malloc(ie*je);

    //

    int gbytes = ie*je;

    addCPUCount(gbytes,"AllocateMemory (uint8_t *):");

    //


    if (plottingInterval == 0)
    {
        plottingInterval = 2;

        // Checks the existance of the
        struct stat st = {0};
        if (stat(outputFolder, &st) == -1)
        {
            mkdir(outputFolder, 0700);
        }

        int iValue;
        //int  count;
        double minimumValue, maximumValue, temporary, **hzlocal;
        minimumValue = -0.1;
        maximumValue = 0.1;

        scaleValue = 256.0 / (maximumValue - minimumValue);
        hzlocal = hz;


        //count = 0;
        #pragma omp parallel for private(i, j, temporary, iValue) shared(hzlocal, gif_frame)
        for (j = 0; j < je; j++)
        {
            for (i = 0; i < ie; i++)
            {
                temporary = hzlocal[i][j];
                temporary = (temporary - minimumValue) * scaleValue;
                iValue = (int)(temporary);
                if (iValue < 0)
                {
                    iValue = 0;
                } /* if */
                if (iValue > 255)
                {
                    iValue = 255;
                } /* if */

                int pos;
                pos = ((i) * je) + j;

                //#pragma omp critical // Seems to speed the overall mehod time by 1+ seconds. Could be due to cache.
                gif_frame[pos] = iValue;
            }
        } /* yForLoop */
    } /* if */
    plottingInterval--;
    
    return gif_frame;
}

// OpenCL

static double *
TransformPointerToVector(
    int imax, int jmax,
    double **src
)
{
    int i, j;
    double *dst;
    // printf("Start Loop\n");
    dst = (double *) malloc((imax*jmax) * sizeof(double));

    //

    int dbytes = (imax*jmax) * sizeof(double);

    addCPUCount(dbytes,"AllocateMemory (double *):");

    //


    for(i = 0; i < imax; i++)
    {
        for(j = 0; j < jmax; j++)
        {
            dst[(i*jmax)+j] = src[i][j];
        }
    }
    // printf("End Loop\n");
    FreeMemory(imax, src);
    return(dst);
}

static void
TimeSteppingLoopUpdateEXPML_OCL(
    int ie, int iebc, int iefbc, int je, int jebc, int jb,
    double *caex, double *cbex,
    double **caexbcb, double **cbexbcb, double **caexbcf, double **cbexbcf,
    double **caexbcl, double **cbexbcl, double **caexbcr, double **cbexbcr,
    double *ex, double **exbcf, double **exbcl, double **exbcr, double **exbcb,
    double *hz, double **hzxbcb, double **hzybcb, double **hzxbcf, double **hzybcf,
    double **hzxbcl, double **hzybcl, double **hzxbcr, double **hzybcr
)
{
    // FRONT
    TimeSteppingLoopUpdateEXPMLFront_OCL(ie, iebc, iefbc, jebc, je, jb, caex, cbex, caexbcf, cbexbcf, ex, exbcf, hz, hzxbcf, hzybcf);
    // BACK
    TimeSteppingLoopUpdateEXPMLBack_OCL(ie, iebc, je, iefbc, jebc, jb, caex, cbex, caexbcb, cbexbcb, ex, exbcb, hz, hzxbcb, hzybcb);
    // LEFT
    TimeSteppingLoopUpdateEXPMLLeft(ie, iebc, iefbc, je, jebc, caexbcl, cbexbcl, exbcl, hzxbcb, hzybcb, hzxbcf, hzybcf, hzxbcl, hzybcl);
    // RIGHT
    TimeSteppingLoopUpdateEXPMLRight(ie, iebc, iefbc, je, jebc, caexbcr, cbexbcr, exbcr, hzxbcb, hzybcb, hzxbcf, hzybcf, hzxbcr, hzybcr);
}

static void
TimeSteppingLoopUpdateEXPMLFront_OCL(
    int ie, int iebc, int iefbc, int jebc, int je, int jb,
    double *caex, double *cbex, double **caexbcf, double **cbexbcf,
    double *ex, double **exbcf,
    double *hz, double **hzxbcf, double **hzybcf
)
{
    int i, j;
    for (i = 0; i < iefbc; i++)
    {
        for (j = 1; j < jebc; j++)
        { // don't Evaluate exbcf at j=0, as it is the PEC
            // note: sign change in the second term from main grid!! ... (due to the Exponential time stepping algorithm?)
            exbcf[i][j] = caexbcf[i][j] * exbcf[i][j] - cbexbcf[i][j] * (hzxbcf[i][j - 1] + hzybcf[i][j - 1] - hzxbcf[i][j] - hzybcf[i][j]);
        } /* jForLoop */
    }     /* iForLoop */

    int pos;
    for (j = 0, i = 0; i < ie; i++)
    { // fill in the edge for ex at j=0 main grid  (ties in the pml with the main grid)
        pos = (i*jb)+j;
        ex[pos] = caex[pos] * ex[pos] - cbex[pos] * (hzxbcf[iebc + i][jebc - 1] + hzybcf[iebc + i][jebc - 1] - hz[(i*je)+j]);
    } /* iForLoop */
}

static void
TimeSteppingLoopUpdateEXPMLBack_OCL(
    int ie, int iebc, int je, int iefbc, int jebc, int jb,
    double *caex, double *cbex, double **caexbcb, double **cbexbcb,
    double *ex, double **exbcb,
    double *hz, double **hzxbcb, double **hzybcb
)
{
    int i, j;
    // Locals

    for (i = 0; i < iefbc; i++)
    {
        for (j = 1; j < jebc; j++)
        { // don't Evaluate exbcb at j=jebc, as it is the PEC, also dont eval at j=0 as this point is the same as j=je on the main grid
            exbcb[i][j] = caexbcb[i][j] * exbcb[i][j] - cbexbcb[i][j] * (hzxbcb[i][j - 1] + hzybcb[i][j - 1] - hzxbcb[i][j] - hzybcb[i][j]);
        } /* jForLoop */
    }     /* iForLoop */

    j = je;
    for (i = 0; i < ie; i++)
    { // fill in the edge for ex at j=je main grid  (ties in the pml with the main grid)
        ex[(i*jb)+j] = caex[(i*jb)+j] * ex[(i*jb)+j] - cbex[(i*jb)+j] * (hz[(i*je)+j - 1] - hzxbcb[iebc + i][0] - hzybcb[iebc + i][0]);
    } /* iForLoop */
}

static void
TimeSteppingLoopUpdateEYPML_OCL(
    int ie, int iebc, int iefbc, int je, int jebc,
    double *caey, double **caeybcb, double **cbeybcb, double **caeybcf, double **cbeybcf,
    double **caeybcl, double **cbeybcl, double **caeybcr, double **cbeybcr,
    double *cbey, double *ey, double **eybcb, double **eybcf, double **eybcl, double **eybcr,
    double *hz, double **hzxbcb, double **hzybcb, double **hzxbcf, double **hzybcf,
    double **hzxbcl, double **hzybcl, double **hzxbcr, double **hzybcr
)
{
    // FRONT
    TimeSteppingLoopUpdateEYPMLFront(iefbc, jebc, caeybcf, cbeybcf, eybcf, hzxbcf, hzybcf);
    // BACK
    TimeSteppingLoopUpdateEYPMLBack(iefbc, jebc, caeybcb, cbeybcb, eybcb, hzxbcb, hzybcb);
    // LEFT
    TimeSteppingLoopUpdateEYPMLLeft_OCL(iebc, je, caey, caeybcl, cbeybcl, cbey, ey, eybcl, hz, hzxbcl, hzybcl);
    // RIGHT
    TimeSteppingLoopUpdateEYPMLRight_OCL(ie, iebc, je, caey, caeybcr, cbeybcr, cbey, ey, eybcr, hz, hzxbcr, hzybcr);
}

static void
TimeSteppingLoopUpdateEYPMLLeft_OCL(
    int iebc, int je,
    double *caey, double **caeybcl, double **cbeybcl,
    double *cbey, double *ey, double **eybcl,
    double *hz, double **hzxbcl, double **hzybcl
)
{
    int i, j;
    // Locals

    for (i = 1; i < iebc; i++)
    { // don't Evaluate eybcb at i=0, as it is the PEC
        for (j = 0; j < je; j++)
        {
            eybcl[i][j] = caeybcl[i][j] * eybcl[i][j] - cbeybcl[i][j] * (hzxbcl[i][j] + hzybcl[i][j] - hzxbcl[i - 1][j] - hzybcl[i - 1][j]);
        } /* jForLoop */
    }     /* iForLoop */

    i = 0;
    for (j = 0; j < je; j++)
    { // fill in the edge for ey at i=0 main grid  (ties in the pml with the main grid)
        ey[(i*je)+j] = caey[(i*je)+j] * ey[(i*je)+j] - cbey[(i*je)+j] * (hz[(i*je)+j] - hzxbcl[iebc - 1][j] - hzybcl[iebc - 1][j]);
    } /* jForLoop */
}

static void
TimeSteppingLoopUpdateEYPMLRight_OCL(
    int ie, int iebc, int je,
    double *caey, double **caeybcr, double **cbeybcr,
    double *cbey, double *ey, double **eybcr,
    double *hz, double **hzxbcr, double **hzybcr
)
{
    int i, j;
    // Locals

    for (i = 1; i < iebc; i++)
    { // don't Evaluate eybcr at i=iebc, as it is the PEC, also dont eval at i=0 as this point is the same as i=ie on the main grid
        for (j = 0; j < je; j++)
        {
            eybcr[i][j] = caeybcr[i][j] * eybcr[i][j] - cbeybcr[i][j] * (hzxbcr[i][j] + hzybcr[i][j] - hzxbcr[i - 1][j] - hzybcr[i - 1][j]);
        } /* jForLoop */
    }     /* iForLoop */

    i = ie;
    for (j = 0; j < je; j++)
    { // fill in the edge for ey at i=ie main grid  (ties in the pml with the main grid)
        ey[(i*je)+j] = caey[(i*je)+j] * ey[(i*je)+j] - cbey[(i*je)+j] * (hzxbcr[0][j] + hzybcr[0][j] - hz[((i-1)*je)+j]);// (i*jmax)+j
    } /* jForLoop */
}

static void
TimeSteppingLoopUpdateHZXPML_OCL(
    int ie, int iebc, int iefbc, int je, int jebc,
    double **dahzxbcb, double **dbhzxbcb, double **dahzxbcf, double **dbhzxbcf,
    double **dahzxbcl, double **dbhzxbcl, double **dahzxbcr, double **dbhzxbcr,
    double *ey, double **eybcb, double **eybcf, double **eybcl, double **eybcr,
    double **hzxbcb, double **hzxbcf, double **hzxbcl, double **hzxbcr
)
{
    // FRONT
    TimeSteppingLoopUpdateHZXPMLFront(iefbc, jebc, dahzxbcf, dbhzxbcf, eybcf, hzxbcf);
    // BACK
    TimeSteppingLoopUpdateHZXPMLBack(iefbc, jebc, dahzxbcb, dbhzxbcb, eybcb, hzxbcb);
    // LEFT
    TimeSteppingLoopUpdateHZXPMLLeft_OCL(je, iebc, dahzxbcl, dbhzxbcl, ey, eybcl, hzxbcl);
    // RIGHT
    TimeSteppingLoopUpdateHZXPMLRight_OCL(ie, je, iebc, dahzxbcr, dbhzxbcr, ey, eybcr, hzxbcr);
}

static void
TimeSteppingLoopUpdateHZXPMLLeft_OCL(
    int je, int iebc,
    double **dahzxbcl, double **dbhzxbcl,
    double *ey, double **eybcl, double **hzxbcl
)
{
    int i, j;
    for (i = 0; i < (iebc - 1); i++)
    { // don't evaluate hzxbcl at i=iebc-1 as it needs ey from main grid, see below
        for (j = 0; j < je; j++)
        {
            hzxbcl[i][j] = dahzxbcl[i][j] * hzxbcl[i][j] - dbhzxbcl[i][j] * (eybcl[i + 1][j] - eybcl[i][j]);
        } /* jForLoop */
    }     /* iForLoop */

    for (i = (iebc - 1), j = 0; j < je; j++)
    { // fix-up hzxbcl at i=iebc-1
        hzxbcl[i][j] = dahzxbcl[i][j] * hzxbcl[i][j] - dbhzxbcl[i][j] * (ey[j] - eybcl[i][j]);
    } /* jForLoop */
}

static void
TimeSteppingLoopUpdateHZXPMLRight_OCL(
    int ie, int je, int iebc,
    double **dahzxbcr, double **dbhzxbcr,
    double *ey, double **eybcr, double **hzxbcr
)
{
    int i, j;
    for (i = 1; i < iebc; i++)
    { // don't evaluate hzxbcl at i=0 as it needs ey from main grid, see below
        for (j = 0; j < je; j++)
        {
            hzxbcr[i][j] = dahzxbcr[i][j] * hzxbcr[i][j] - dbhzxbcr[i][j] * (eybcr[i + 1][j] - eybcr[i][j]);
        } /* jForLoop */
    }     /* iForLoop */

    for (i = 0, j = 0; j < je; j++)
    { // fix-up hzxbcl at i=0
        hzxbcr[i][j] = dahzxbcr[i][j] * hzxbcr[i][j] - dbhzxbcr[i][j] * (eybcr[i + 1][j] - ey[(ie*je)+j]);
    } /* jForLoop */
}

static void
TimeSteppingLoopUpdateHZYPML_OCL(
    int ie, int iebc, int iefbc, int je, int jebc,int jb,
    double **dahzybcb, double **dbhzybcb, double **dahzxbcf, double **dbhzxbcf, double **dahzybcf,
    double **dbhzybcf, double **dahzybcl, double **dbhzybcl, double **dahzybcr, double **dbhzybcr,
    double *ex, double **exbcb, double **exbcf, double **exbcl, double **exbcr, double **hzybcb,
    double **hzybcf, double **hzybcl, double **hzybcr
)
{
    // FRONT
    TimeSteppingLoopUpdateHZYPMLFront_OCL(ie, iebc, iefbc, je, jebc, jb, dahzxbcf, dbhzxbcf, dahzybcf, dbhzybcf, ex, exbcf, exbcl, exbcr, hzybcf);
    // BACK
    TimeSteppingLoopUpdateHZYPMLBack_OCL(ie, iebc, iefbc, je, jebc, jb, dahzybcb, dbhzybcb, ex, exbcb, exbcl, exbcr, hzybcb);
    // LEFT
    TimeSteppingLoopUpdateHZYPMLLeft(ie, iebc, je, jebc, dahzybcl, dbhzybcl, exbcl, hzybcl);
    // RIGHT
    TimeSteppingLoopUpdateHZYPMLRight(iebc, je, dahzybcr, dbhzybcr, exbcr, hzybcr);
}

static void
TimeSteppingLoopUpdateHZYPMLFront_OCL(
    int ie, int iebc, int iefbc, int je, int jebc, int jb,
    double **dahzxbcf, double **dbhzxbcf, double **dahzybcf, double **dbhzybcf,
    double *ex, double **exbcf, double **exbcl, double **exbcr, double **hzybcf
)
{
    int i, j;
    for (i = 0; i < iefbc; i++)
    {
        for (j = 0; j < (jebc - 1); j++)
        { // don't evaluate hzxbcf at j=jebc-1 as it needs data from main,left,right grids, see below
            // note: sign change in the second term from main grid!!
            hzybcf[i][j] = dahzybcf[i][j] * hzybcf[i][j] - dbhzybcf[i][j] * (exbcf[i][j] - exbcf[i][j + 1]);
        } /* jForLoop */
    }     /* iForLoop */

    for (j = (jebc - 1), i = 0; i < iebc; i++)
    { // fix-up hzybcf at j=jebc-1, with left grid
        hzybcf[i][j] = dahzybcf[i][j] * hzybcf[i][j] - dbhzybcf[i][j] * (exbcf[i][j] - exbcl[i][0]);
    } /* iForLoop */

    for (j = (jebc - 1), i = 0; i < ie; i++)
    { // fix-up hzybcf at j=jebc-1, with main grid
        hzybcf[iebc + i][j] = dahzybcf[iebc + i][j] * hzybcf[iebc + i][j] - dbhzybcf[iebc + i][j] * (exbcf[iebc + i][j] - ex[(i*jb)]); // (i*jmax)+j
    } /* iForLoop */

    for (j = (jebc - 1), i = 0; i < iebc; i++)
    { // fix-up hzybcf at j=jebc-1, with right grid
        hzybcf[iebc + ie + i][j] = dahzybcf[iebc + ie + i][j] * hzybcf[iebc + ie + i][j] - dbhzybcf[iebc + ie + i][j] * (exbcf[iebc + ie + i][j] - exbcr[i][0]);
    } /* iForLoop */
}

static void
TimeSteppingLoopUpdateHZYPMLBack_OCL(
    int ie, int iebc, int iefbc, int je, int jebc, int jb,
    double **dahzybcb, double **dbhzybcb,
    double *ex, double **exbcb, double **exbcl, double **exbcr, double **hzybcb
)
{
    int i, j;
    for (i = 0; i < iefbc; i++)
    {
        for (j = 1; j < jebc; j++)
        { // don't evaluate hzxbcb at j=0 as it needs data from main,left,right grids, see below
            hzybcb[i][j] = dahzybcb[i][j] * hzybcb[i][j] - dbhzybcb[i][j] * (exbcb[i][j] - exbcb[i][j + 1]);
        } /* jForLoop */
    }     /* iForLoop */

    for (j = 0, i = 0; i < iebc; i++)
    { // fix-up hzybcb at j=0, with left grid
        hzybcb[i][j] = dahzybcb[i][j] * hzybcb[i][j] - dbhzybcb[i][j] * (exbcl[i][je] - exbcb[i][j + 1]);
    } /* iForLoop */

    for (j = 0, i = 0; i < ie; i++)
    { // fix-up hzybcb at j=0, with main grid
        hzybcb[iebc + i][j] = dahzybcb[iebc + i][j] * hzybcb[iebc + i][j] - dbhzybcb[iebc + i][j] * (ex[(i*jb)+je] - exbcb[iebc + i][j + 1]); //(i*jmax)+j
    } /* iForLoop */

    for (j = 0, i = 0; i < iebc; i++)
    { // fix-up hzybcb at j=0, with right grid
        hzybcb[iebc + ie + i][j] = dahzybcb[iebc + ie + i][j] * hzybcb[iebc + ie + i][j] - dbhzybcb[iebc + ie + i][j] * (exbcr[i][je] - exbcb[iebc + ie + i][j + 1]);
    } /* iForLoop */
}

static void
TimeSteppingLoopUpdateEXEYMain_OCL(
    int ie, int je, int ib, int jb,
    double *caex, double *cbex, double *caey, double *cbey, double *ex, double *ey, double *hz,
    cl_context context,
    cl_program program,
    cl_command_queue queue,
    cl_mem cl_hz, cl_mem cl_ex, cl_mem cl_ey, cl_mem cl_caex, cl_mem cl_cbex, cl_mem cl_caey, cl_mem cl_cbey
)
{
    cl_int ret, cl_ie, cl_je, cl_jb;
    cl_ie = ie; cl_je = je; cl_jb = jb;

    // Create kernals for loops
    cl_kernel loop_one = clCreateKernel(program,
                                        "TimeSteppingLoopUpdateEXEYMain_loop_one",
                                        &ret);

    cl_kernel loop_two = clCreateKernel(program,
                                        "TimeSteppingLoopUpdateEXEYMain_loop_two",
                                        &ret);

    // Assign variables to buffers
    ret =   clEnqueueWriteBuffer(
                queue,
                cl_ex,
                CL_TRUE,
                0,
                ie * jb * sizeof(double),
                &(ex[0]),
                0, NULL, NULL
            );

    ret =   clEnqueueWriteBuffer(
                queue,
                cl_ey,
                CL_TRUE,
                0,
                ib * je * sizeof(double),
                &(ey[0]),
                0, NULL, NULL
            );

    ret =   clEnqueueWriteBuffer(
                queue,
                cl_hz,
                CL_TRUE,
                0,
                ie * je * sizeof(double),
                &(hz[0]),
                0, NULL, NULL
            );

    // Loop One Kernal Args
    ret = clSetKernelArg(loop_one, 0, sizeof(cl_mem), (void*) &cl_ex); // ex
    ret = clSetKernelArg(loop_one, 1, sizeof(cl_mem), (void*) &cl_caex); //caex
    ret = clSetKernelArg(loop_one, 2, sizeof(cl_mem), (void*) &cl_cbex); // cbex
    ret = clSetKernelArg(loop_one, 3, sizeof(cl_mem), (void*) &cl_hz); // hz
    ret = clSetKernelArg(loop_one, 4, sizeof(cl_int), &cl_je); // je
    ret = clSetKernelArg(loop_one, 5, sizeof(cl_int), &cl_jb); // ie

    // Loop Two Kernal Args
    ret = clSetKernelArg(loop_two, 0, sizeof(cl_mem), (void*) &cl_ey); // ey
    ret = clSetKernelArg(loop_two, 1, sizeof(cl_mem), (void*) &cl_caey); //caey
    ret = clSetKernelArg(loop_two, 2, sizeof(cl_mem), (void*) &cl_cbey); // cbey
    ret = clSetKernelArg(loop_two, 3, sizeof(cl_mem), (void*) &cl_hz); // hz
    ret = clSetKernelArg(loop_two, 4, sizeof(cl_int), &cl_je); // je
    ret = clSetKernelArg(loop_two, 5, sizeof(cl_int), &cl_ie); // ie

    size_t work_size_loop_one = (size_t)ie;

    ret =   clEnqueueNDRangeKernel(
                queue,
                loop_one,
                1,
                NULL,
                &work_size_loop_one,
                NULL, 0, NULL, NULL
            );

    // ret = clFlush(queue);
    // ret = clFinish(queue);

    // End of Loop One

    size_t work_size_loop_two = (size_t)(ie - 1);

    ret =   clEnqueueNDRangeKernel(
                queue,
                loop_two,
                1,
                NULL,
                &work_size_loop_two,
                NULL, 0, NULL, NULL
            );

    ret = clFlush(queue);
    ret = clFinish(queue);

    ret =   clEnqueueReadBuffer(
                queue,
                cl_ex,
                CL_TRUE,
                0,
                ie * jb * sizeof(double),
                &(ex[0]),
                0, NULL, NULL
            );

    ret =   clEnqueueReadBuffer(
                queue,
                cl_ey,
                CL_TRUE,
                0,
                ib * je * sizeof(double),
                &(ey[0]),
                0, NULL, NULL
            );

    ret = clReleaseKernel(loop_one);
    ret = clReleaseKernel(loop_two);
}

static void
TimeSteppingLoopUpdateMagneticFieldHZ_OCL(
    int ie, int is, int je, int js, int n, int ib, int jb,
    double source[],
    double *dahz, double *dbhz, double *ex, double *ey, double *hz,
    cl_context context,
    cl_program program,
    cl_command_queue queue,
    cl_mem cl_hz, cl_mem cl_ex, cl_mem cl_ey, cl_mem cl_dahz, cl_mem cl_dbhz
    )
{
    cl_int ret, cl_ie, cl_je, cl_jb;
    cl_ie = ie; cl_je = je; cl_jb = jb;

    // Create kernals for loops
    cl_kernel TimeSteppingLoopUpdateMagneticFieldHZ = clCreateKernel(program,
                                                                    "TimeSteppingLoopUpdateMagneticFieldHZ",
                                                                    NULL );

    // Assign variables to buffers

    clEnqueueWriteBuffer(
        queue,
        cl_hz,
        CL_TRUE,
        0,
        ie * je * sizeof(double),
        &(hz[0]),
        0, NULL, NULL
    );

    clEnqueueWriteBuffer(
        queue,
        cl_ex,
        CL_TRUE,
        0,
        ib * je * sizeof(double),
        &(ex[0]),
        0, NULL, NULL
    );

    clEnqueueWriteBuffer(
        queue,
        cl_ey,
        CL_TRUE,
        0,
        ib * je * sizeof(double),
        &(ey[0]),
        0, NULL, NULL
    );

    clSetKernelArg(TimeSteppingLoopUpdateMagneticFieldHZ, 0, sizeof(cl_mem), (void*) &cl_hz); // hz
    clSetKernelArg(TimeSteppingLoopUpdateMagneticFieldHZ, 1, sizeof(cl_mem), (void*) &cl_dahz); //dahz
    clSetKernelArg(TimeSteppingLoopUpdateMagneticFieldHZ, 2, sizeof(cl_mem), (void*) &cl_dbhz); // dbhz
    clSetKernelArg(TimeSteppingLoopUpdateMagneticFieldHZ, 3, sizeof(cl_mem), (void*) &cl_ex); // ex
    clSetKernelArg(TimeSteppingLoopUpdateMagneticFieldHZ, 4, sizeof(cl_mem), (void*) &cl_ey); // ey
    clSetKernelArg(TimeSteppingLoopUpdateMagneticFieldHZ, 5, sizeof(cl_int), &cl_je); // je
    clSetKernelArg(TimeSteppingLoopUpdateMagneticFieldHZ, 6, sizeof(cl_int), &cl_ie); // ie
    clSetKernelArg(TimeSteppingLoopUpdateMagneticFieldHZ, 7, sizeof(cl_int), &cl_jb); // ie

    size_t global_work_size = (size_t)ie;

    clEnqueueNDRangeKernel(
        queue,
        TimeSteppingLoopUpdateMagneticFieldHZ,
        1,
        NULL,
        &global_work_size,
        NULL, 0, NULL, NULL
    );

    clFlush(queue);
    clFinish(queue);

    clEnqueueReadBuffer(
        queue,
        cl_hz,
        CL_TRUE,
        0,
        ie * je * sizeof(double),
        &(hz[0]),
        0, NULL, NULL
    );

    clReleaseKernel(TimeSteppingLoopUpdateMagneticFieldHZ);

    hz[(is * ie) + js] = source[n];
}


static uint8_t *TimeSteppingLoopPlotFields_OCL(
    int centery, int ie, int je, int n, int plottingInterval,
    double minimumValue, double maximumValue, double scaleValue,
    char filename[], char outputFolder[],
    double *ex, double *ey, double *hz,
    cl_context context,
    cl_program program,
    cl_command_queue queue,
    cl_mem cl_gif_frame, cl_mem cl_hz
    )
{
    //int i, j;
    uint8_t *gif_frame = (uint8_t *) malloc(ie*je);

    //

    int fbytes = ie*je;

    addCPUCount(fbytes,"AllocateMemory (uint8_t *):");

    //


    if (plottingInterval == 0)
    {

        plottingInterval = 2;

        // Checks the existance of the
        struct stat st = {0};
        if (stat(outputFolder, &st) == -1)
        {
            mkdir(outputFolder, 0700);
        }

        //int iValue;
        //int  count;
        double minimumValue, maximumValue;
        //double temporary;
        minimumValue = -0.1;
        maximumValue = 0.1;

        scaleValue = 256.0 / (maximumValue - minimumValue);

        //count = 0;

        // Create kernals for loops

        cl_kernel TimeSteppingLoopPlotFields_OCL = clCreateKernel(program,
                                                                "TimeSteppingLoopPlotFields_OCL",
                                                                NULL );
        
        clEnqueueWriteBuffer(
            queue,
            cl_hz,
            CL_TRUE,
            0,
            ie * je * sizeof(double),
            &(hz[0]),
            0,
            NULL, NULL
        );


        clSetKernelArg(TimeSteppingLoopPlotFields_OCL, 0, sizeof(cl_mem), (void*) &cl_gif_frame); // gif_frame
        clSetKernelArg(TimeSteppingLoopPlotFields_OCL, 1, sizeof(cl_mem), (void*) &cl_hz); // hz
        clSetKernelArg(TimeSteppingLoopPlotFields_OCL, 2, sizeof(minimumValue), &minimumValue); // minimumValue
        clSetKernelArg(TimeSteppingLoopPlotFields_OCL, 3, sizeof(scaleValue), &scaleValue); // scaleValue
        clSetKernelArg(TimeSteppingLoopPlotFields_OCL, 4, sizeof(ie), &ie); // ie
        clSetKernelArg(TimeSteppingLoopPlotFields_OCL, 5, sizeof(je), &je); // ie

        size_t global_work_size = (size_t)je;

        clEnqueueNDRangeKernel(
            queue,
            TimeSteppingLoopPlotFields_OCL,
            1,
            NULL,
            &global_work_size,
            NULL, 0, NULL, NULL
        );

        clFlush(queue);
        clFinish(queue);

        clEnqueueReadBuffer(
            queue,
            cl_gif_frame,
            CL_TRUE,
            0,
            (ie * je) * sizeof(uint8_t),
            &(gif_frame[0]),
            0,
            NULL, NULL
        );

        clReleaseKernel(TimeSteppingLoopPlotFields_OCL);
    }
    plottingInterval--;

    return gif_frame;
}

void
FreeMemoryUsages(
    int ie, int ib, int iefbc, int ibfbc, int iebc, int ibbc,
    double **eybcf, double **exbcf, double **hzxbcf, double **hzybcf,
    double **exbcb, double **eybcb, double **hzxbcb, double **hzybcb,
    double **exbcl, double **eybcl, double **hzxbcl, double **hzybcl,
    double **exbcr, double **eybcr, double **hzxbcr, double **hzybcr,
    double **caexbcf, double **cbexbcf, double **caexbcb, double **cbexbcb,
    double **caexbcl, double **cbexbcl, double **caexbcr, double **cbexbcr,
    double **caeybcf, double **cbeybcf, double **caeybcb, double **cbeybcb,
    double **caeybcl, double **cbeybcl, double **caeybcr, double **cbeybcr,
    double **dahzxbcf, double **dbhzxbcf, double **dahzxbcb, double **dbhzxbcb,
    double **dahzxbcl, double **dbhzxbcl, double **dahzxbcr, double **dbhzxbcr,
    double **dahzybcf, double **dbhzybcf, double **dahzybcb, double **dbhzybcb,
    double **dahzybcl, double **dbhzybcl, double **dahzybcr, double **dbhzybcr
)
{
    FreeMemory(iefbc, exbcf);
    FreeMemory(ibfbc, eybcf);
    FreeMemory(iefbc, hzxbcf);
    FreeMemory(iefbc, hzybcf);

    FreeMemory(iefbc, exbcb);
    FreeMemory(ibfbc, eybcb);
    FreeMemory(iefbc, hzxbcb);
    FreeMemory(iefbc, hzybcb);

    FreeMemory(iebc, exbcl);
    FreeMemory(iebc, eybcl);
    FreeMemory(iebc, hzxbcl);
    FreeMemory(iebc, hzybcl);

    FreeMemory(iebc, exbcr);
    FreeMemory(ibbc, eybcr);
    FreeMemory(iebc, hzxbcr);
    FreeMemory(iebc, hzybcr);

    FreeMemory(iefbc, caexbcf);
    FreeMemory(iefbc, cbexbcf);
    FreeMemory(iefbc, caexbcb);
    FreeMemory(iefbc, cbexbcb);

    FreeMemory(iebc, caexbcl);
    FreeMemory(iebc, cbexbcl);
    FreeMemory(iebc, caexbcr);
    FreeMemory(iebc, cbexbcr);

    FreeMemory(ibfbc, caeybcf);
    FreeMemory(ibfbc, cbeybcf);
    FreeMemory(ibfbc, caeybcb);
    FreeMemory(ibfbc, cbeybcb);

    FreeMemory(iebc, caeybcl);
    FreeMemory(iebc, cbeybcl);
    FreeMemory(iebc, caeybcr);
    FreeMemory(iebc, cbeybcr);

    FreeMemory(iefbc, dahzxbcf);
    FreeMemory(iefbc, dbhzxbcf);
    FreeMemory(iefbc, dahzxbcb);
    FreeMemory(iefbc, dbhzxbcb);

    FreeMemory(iebc, dahzxbcl);
    FreeMemory(iebc, dbhzxbcl);
    FreeMemory(iebc, dahzxbcr);
    FreeMemory(iebc, dbhzxbcr);

    FreeMemory(iefbc, dahzybcf);
    FreeMemory(iefbc, dbhzybcf);
    FreeMemory(iefbc, dahzybcb);
    FreeMemory(iefbc, dbhzybcb);

    FreeMemory(iebc, dahzybcl);
    FreeMemory(iebc, dbhzybcl);
    FreeMemory(iebc, dahzybcr);
    FreeMemory(iebc, dbhzybcr);
}

