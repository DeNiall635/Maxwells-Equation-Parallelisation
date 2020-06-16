#ifndef YEE2D
#define YEE2D

// OPENCL LIBRARY
//#include <CL/cl.h>

#include <CL/opencl.h>

//#include "gifenc.h"

enum RunningMode { 
    Standard = 0, 
    OpenMP = 1, 
    OpenCL = 2/*,
    MPI = 3*/
};

//int main();

// EVALUATE FDTD
void EvaluateFdtd(
    enum RunningMode runningMode,
    int xRegion, int yRegion, int debugMode
);


// ALLOCATE MEMORY
static double **AllocateMemory(
    int imax, int jmax,
    double initialValue
);


// FREE MEMORY
static void FreeMemory(
    int imax,
    double **pointer
);


// INITIALISE VARIABLES
static void InitialiseSource(
    int nmax,
    double source[]
);

static void WaveExcitation(
    double tau, double omega, double delay, double dt,
    double source[]
);

static void UpdateCoefficients(
    int media,
    double dt, double dx, double eaf, double haf, double epsz, double muz,
    double *sig, double *eps, double *ca, double *cb, double *da, double *db,  double *sim, double *mur
);

static void AddMetalCylinder(
    int icenter, int jcenter, int ie, int je,
    double rad, double temporaryi, double temporaryj, double dist2,
    double *ca, double *cb, double **caex, double **cbex, double **caey, double **cbey
);


// FRONT REGION
static void FrontRegion(
    int ie, int iebc, int iefbc, int ibfbc, int jebc,
    double cb1, double da1, double db1, double muz, double epsz, double ca1, double orderbc, double bcfactor, double sigmay, double sigmays, double dt, double dx, double y1, double y2,
    double ca[], double cb[], double da[], double db[], double eps[],
    double **cbex, double **caex, double **cbexbcf, double **caexbcf, double **caexbcl, double **cbexbcl, double **caexbcr,
    double **cbexbcr, double **caeybcf, double **cbeybcf, double **dahzybcf, double **dbhzybcf, double **dahzxbcf, double **dbhzxbcf
);

static void FrontRegionFirstLoop(
    int iefbc,
    double **caexbcf, double **cbexbcf
);

static void FrontRegionSecondLoop(
    int iefbc, int jebc,
    double bcfactor, double ca1, double cb1, double dt, double dx, double epsz, double orderbc, double sigmay, double y1, double y2,
    double eps[], double **caexbcf, double **cbexbcf
);

static void FrontRegionThirdLoop(
    int ie,
    double bcfactor, double ca1, double cb1, double dt, double dx, double epsz, double orderbc, double sigmay,
    double eps[],
    double **caex, double **cbex
);

static void FrontRegionFourthLoop(
    int iebc,
    double ca1, double cb1,
    double **caex, double **cbex, double **caexbcl, double **cbexbcl, double **caexbcr, double **cbexbcr
);

static void FrontRegionFifthLoop(
    int ibfbc, int iefbc, int jebc,
    double bcfactor, double da1, double db1, double dt, double dx, double epsz, double muz, double orderbc, double sigmay, double sigmays, double y1, double y2,
    double ca[], double cb[], double da[], double db[], double eps[],
    double **caeybcf, double **cbeybcf, double **dahzybcf, double **dbhzybcf, double **dahzxbcf, double **dbhzxbcf
);


// BACK REGION
static void BackRegion(
    int ie, int je, int ibfbc, int iebc, int iefbc, int jebc,
    double bcfactor, double ca1, double cb1, double da1, double db1, double dt, double dx, double epsz, double muz, double orderbc, double sigmay, double sigmays, double y1, double y2,
    double ca[], double cb[], double da[], double db[], double eps[],
    double **caex, double **cbex, double **caexbcb, double **cbexbcb, double **caeybcb, double **cbeybcb, double **caexbcl, double **cbexbcl,
    double **caexbcr, double **cbexbcr, double **dahzybcb, double **dbhzybcb, double **dahzxbcb, double **dbhzxbcb
);

static void BackRegionFirstLoop(
    int jebc, int iefbc,
    double **caexbcb, double **cbexbcb
);

static void BackRegionSecondLoop(
    int iefbc, int jebc,
    double bcfactor, double ca1, double cb1, double dt, double dx, double epsz, double orderbc, double sigmay, double y1, double y2,
    double eps[],
    double **caexbcb, double **cbexbcb
);

static void BackRegionThirdLoop(
    int ie, int je,
    double bcfactor, double ca1, double cb1, double dt, double dx, double epsz, double orderbc, double sigmay,
    double eps[],
    double **caex, double **cbex
);

static void BackRegionFourthLoop(
    int je, int iebc,
    double ca1, double cb1,
    double **caexbcl, double **cbexbcl, double **caexbcr, double **cbexbcr
);

static void BackRegionFifthLoop(
    int ibfbc, int iefbc, int jebc,
    double bcfactor, double da1, double db1, double dt, double dx, double epsz, double muz, double orderbc, double sigmay, double sigmays, double y1, double y2,
    double ca[], double cb[], double da[], double db[], double eps[],
    double **caeybcb, double **cbeybcb, double **dahzybcb, double **dbhzybcb, double **dahzxbcb, double **dbhzxbcb
);


// LEFT REGION
static void LeftRegion(
    int je, int iebc, int jebc,
    double bcfactor, double ca1, double cb1, double da1, double db1, double dt, double dx, double epsz, double muz, double orderbc, double sigmax, double sigmaxs, double x1, double x2,
    double ca[], double cb[], double da[], double db[], double eps[],
    double **caey, double **cbey, double **caexbcl, double **cbexbcl, double **caeybcl, double **cbeybcl, double **caeybcb, double **cbeybcb, double **caeybcf, double **cbeybcf,
    double **dahzxbcb, double **dbhzxbcb, double **dahzybcf, double **dbhzybcf, double **dahzxbcf, double **dbhzxbcf, double **dahzxbcl, double **dbhzxbcl, double **dahzybcl, double **dbhzybcl
);

static void LeftRegionFirstLoop(
    int je,
    double **caeybcl, double **cbeybcl
);

static void LeftRegionSecondLoop(
    int je, int iebc, int jebc,
    double bcfactor, double ca1, double cb1, double epsz, double x1, double x2, double dt, double dx, double orderbc, double sigmax,
    double eps[],
    double **caeybcl, double **cbeybcl, double **caeybcf, double **cbeybcf, double **caeybcb, double **cbeybcb
);

static void LeftRegionThirdLoop(
    int je, int iebc, int jebc,
    double ca1, double cb1,
    double **caey, double **cbey, double **caeybcf, double **cbeybcf, double **caeybcb, double **cbeybcb
);

static void LeftRegionFourthLoop(
    int je, int iebc, int jebc,
    double bcfactor, double da1, double db1, double dt, double dx, double epsz, double muz, double orderbc, double sigmax, double sigmaxs, double x1, double x2,
    double ca[], double cb[], double da[], double db[], double eps[],
    double **caexbcl, double **cbexbcl, double **dahzybcl, double **dbhzybcl, double **dahzxbcl, double **dbhzxbcl, double **dahzxbcf, double **dbhzxbcf, double **dahzxbcb, double **dbhzxbcb
);


// RIGHT REGION
static void RightRegion(
    int ie, int je, int iebc,
    double bcfactor, double ca1, double cb1, double epsz, double da1, double db1, double dt, double dx, int jebc, double muz, double orderbc, double sigmax, double sigmaxs, double x1, double x2,
    double ca[], double cb[], double da[], double db[], double eps[],
    double **caey, double **cbey, double **caeybcr, double **cbeybcr, double **caeybcf, double **cbeybcf, double **caeybcb, double **cbeybcb, double **caexbcr, double **cbexbcr,
    double **dahzxbcf, double **dbhzxbcf, double **dahzxbcb, double **dbhzxbcb, double **dahzxbcr, double **dbhzxbcr, double **dahzybcr, double **dbhzybcr
);

static void RightRegionFirstLoop(
    int je, int iebc,
    double **caeybcr, double **cbeybcr
);

static void RightRegionSecondLoop(
    int ie, int je, int iebc, int jebc,
    double bcfactor, double ca1, double cb1, double dt, double dx, double epsz, double orderbc, double sigmax, double x1, double x2,
    double eps[],
    double **caeybcr, double **cbeybcr, double **caeybcf, double **cbeybcf, double **caeybcb, double **cbeybcb
);

static void RightRegionThirdLoop(
    int ie, int je, int iebc, int jebc,
    double bcfactor, double ca1, double cb1, double dt, double dx, double epsz, double orderbc, double sigmax,
    double eps[],
    double **caey, double **cbey, double **caeybcb, double **cbeybcb, double **caeybcf, double **cbeybcf
);

static void RightRegionFourthLoop(
    int ie, int je, int iebc, int jebc,
    double bcfactor, double da1, double db1, double dt, double dx, double epsz, double muz, double orderbc, double sigmax, double sigmaxs, double x1, double x2,
    double ca[], double cb[], double da[], double db[], double eps[],
    double **caexbcr, double **cbexbcr, double **dahzxbcf, double **dbhzxbcf, double **dahzxbcb, double **dbhzxbcb, double **dahzxbcr, double **dbhzxbcr, double **dahzybcr, double **dbhzybcr
);


// TIME STEPPING LOOP
static void TimeSteppingLoop(
    int centery, int ie, int iebc, int is, int je, int iefbc, int jebc, int js, int nmax, int plottingInterval, int ib, int jb,
    double minimumValue, double maximumValue, double scaleValue,
    double source[], char filename[], char outputFolder[],
    double **caex, double **cbex, double **caey, double **cbey, double **caexbcb, double **cbexbcb, double **caexbcf, double **cbexbcf, double **caexbcl, double **cbexbcl, double **caexbcr, double **cbexbcr,
    double **caeybcb, double **cbeybcb, double **caeybcf, double **cbeybcf, double **caeybcl, double **cbeybcl, double **caeybcr, double **cbeybcr, double **dahz, double **dbhz, double **dahzxbcb,
    double **dbhzxbcb, double **dahzybcb, double **dbhzybcb, double **dahzxbcf, double **dahzybcf, double **dbhzybcf, double **dbhzxbcf, double **dahzxbcl, double **dbhzxbcl, double **dahzybcl,
    double **dbhzybcl, double **dahzybcr, double **dbhzybcr, double **dahzxbcr, double **dbhzxbcr, double **ex, double **exbcf, double **exbcl, double **exbcr, double **exbcb, double **ey, double **eybcb,
    double **eybcf, double **eybcl, double **eybcr, double **hz, double **hzxbcb, double **hzybcb, double **hzxbcf, double **hzybcf, double **hzxbcl, double **hzybcl, double **hzxbcr, double **hzybcr,
    enum RunningMode runningMode,
    int debugMode
);

static char* buildFilename(char* src, int ie, int je);


// TIME STEPPING LOOP EXPML
static void TimeSteppingLoopUpdateEXPML(
    int ie, int iebc, int iefbc, int je, int jebc,
    double **caex, double **cbex, double **caexbcb, double **cbexbcb, double **caexbcf, double **cbexbcf, double **caexbcl, double **cbexbcl, double **caexbcr, double **cbexbcr, double **ex, double **exbcf,
    double **exbcl, double **exbcr, double **exbcb, double **hz, double **hzxbcb, double **hzybcb, double **hzxbcf, double **hzybcf, double **hzxbcl, double **hzybcl, double **hzxbcr, double **hzybcr
);

static void TimeSteppingLoopUpdateEXPMLFront(
    int ie, int iebc, int iefbc, int jebc, double **caex, double **cbex, double **caexbcf, double **cbexbcf, double **ex, double **exbcf, double **hz, double **hzxbcf, double **hzybcf
);

static void TimeSteppingLoopUpdateEXPMLBack(
    int ie, int iebc, int je, int iefbc, int jebc, double **caex, double **cbex, double **caexbcb, double **cbexbcb, double **ex, double **exbcb, double **hz, double **hzxbcb, double **hzybcb
);

static void TimeSteppingLoopUpdateEXPMLLeft(
    int ie, int iebc, int iefbc, int je, int jebc, double **caexbcl, double **cbexbcl, double **exbcl, double **hzxbcb, double **hzybcb, double **hzxbcf, double **hzybcf, double **hzxbcl, double **hzybcl
);

static void TimeSteppingLoopUpdateEXPMLRight(
    int ie, int iebc, int iefbc, int je, int jebc, double **caexbcr, double **cbexbcr, double **exbcr, double **hzxbcb, double **hzybcb, double **hzxbcf, double **hzybcf, double **hzxbcr, double **hzybcr
);


// TIME STEPPING LOOP EYPML
static void TimeSteppingLoopUpdateEYPML(
    int ie, int iebc, int iefbc, int je, int jebc,
    double **caey, double **caeybcb, double **cbeybcb, double **caeybcf, double **cbeybcf, double **caeybcl, double **cbeybcl, double **caeybcr, double **cbeybcr,double **cbey,double **ey, double **eybcb,
    double **eybcf, double **eybcl, double **eybcr, double **hz, double **hzxbcb, double **hzybcb, double **hzxbcf, double **hzybcf, double **hzxbcl, double **hzybcl, double **hzxbcr, double **hzybcr
);

static void TimeSteppingLoopUpdateEYPMLFront(
    int iefbc, int jebc,
    double **caeybcf, double **cbeybcf, double **eybcf, double **hzxbcf, double **hzybcf
);

static void TimeSteppingLoopUpdateEYPMLBack(
    int iefbc, int jebc,
    double **caeybcb, double **cbeybcb, double **eybcb, double **hzxbcb, double **hzybcb
);

static void TimeSteppingLoopUpdateEYPMLLeft(
    int iebc, int je,
    double **caey, double **caeybcl, double **cbeybcl, double **cbey, double **ey, double **eybcl, double **hz, double **hzxbcl, double **hzybcl
);

static void TimeSteppingLoopUpdateEYPMLRight(
    int ie, int iebc, int je,
    double **caey, double **caeybcr, double **cbeybcr, double **cbey, double **ey, double **eybcr, double **hz, double **hzxbcr, double **hzybcr
);

// REMOVE
// TIME STEPPING LOOP MAGNETIC FIELD HZ
// static void TimeSteppingLoopUpdateMagneticFieldHZ(
//     int ie, int is, int je, int js, int n,
//     double source[],
//     double **dahz, double **dbhz, double **ex, double **ey, double **hz
// );


// TIME STEPPING LOOP HZXPML
static void TimeSteppingLoopUpdateHZXPML(
    int ie, int iebc, int iefbc, int je, int jebc,
    double **dahzxbcb, double **dbhzxbcb, double **dahzxbcf, double **dbhzxbcf, double **dahzxbcl, double **dbhzxbcl, double **dahzxbcr, double **dbhzxbcr,
    double **ey,double **eybcb,double **eybcf, double **eybcl, double **eybcr, double **hzxbcb, double **hzxbcf, double **hzxbcl, double **hzxbcr
);

static void TimeSteppingLoopUpdateHZXPMLFront(
    int iefbc, int jebc,
    double **hzxbcf, double **dahzxbcf, double **dbhzxbcf, double **eybcf
);

static void TimeSteppingLoopUpdateHZXPMLBack(
    int iefbc, int jebc,
    double **hzxbcb, double **dahzxbcb, double **dbhzxbcb, double **eybcb
);

static void TimeSteppingLoopUpdateHZXPMLLeft(
    int je, int iebc,
    double **hzxbcl, double **dahzxbcl, double **dbhzxbcl, double **ey, double **eybcl
);

static void TimeSteppingLoopUpdateHZXPMLRight(
    int ie, int je, int iebc,
    double **hzxbcr, double **dahzxbcr, double **dbhzxbcr, double **ey, double **eybcr
);


// TIME STEPPING LOOP HZYPML
static void TimeSteppingLoopUpdateHZYPML(
    int ie, int iebc, int iefbc, int je, int jebc, double **dahzybcb, double **dbhzybcb, double **dahzxbcf, double **dbhzxbcf, double **dahzybcf,
    double **dbhzybcf, double **dahzybcl, double **dbhzybcl, double **dahzybcr, double **dbhzybcr, double **ex, double **exbcb, double **exbcf, double **exbcl,
    double **exbcr, double **hzybcb, double **hzybcf, double **hzybcl, double **hzybcr
);

static void TimeSteppingLoopUpdateHZYPMLFront(
    int ie, int iebc, int iefbc, int je, int jebc, double **dahzxbcf, double **dbhzxbcf, double **dahzybcf, double **dbhzybcf, double **ex,
    double **exbcf, double **exbcl, double **exbcr, double **hzybcf
);

static void TimeSteppingLoopUpdateHZYPMLBack(
    int ie, int iebc, int iefbc, int je, int jebc, double **dahzybcb, double **dbhzybcb, double **ex, double **exbcb, double **exbcl, double **exbcr, double **hzybcb
);

static void TimeSteppingLoopUpdateHZYPMLLeft(
    int ie, int iebc, int je, int jebc, double **dahzybcl, double **dbhzybcl, double **exbcl, double **hzybcl
);

static void TimeSteppingLoopUpdateHZYPMLRight(
    int iebc, int je, double **dahzybcr, double **dbhzybcr, double **exbcr, double **hzybcr
);

// REMOVE
// TIME STEPPING LOOP PLOT FIELDS / OUTPUT / FRAME BUILDER
// static uint8_t *TimeSteppingLoopPlotFields(
//     int centery, int ie, int je, int n, int plottingInterval,
//     double minimumValue, double maximumValue, double scaleValue,
//     char filename[],
//     double **ex, double **ey, double **hz
// );

// Base Methods

// TIME STEPPING LOOP EXEY
static void TimeSteppingLoopUpdateEXEYMain_STD(
    int ie, int je,
    double **caex, double **cbex, double **caey, double **cbey, double **ex, double **ey, double **hz
);

// TIME STEPPING LOOP HZ
static void TimeSteppingLoopUpdateMagneticFieldHZ_STD(
    int ie, int is, int je, int js, int n,
    double source[],
    double **dahz, double **dbhz, double **ex, double **ey, double **hz
);

// TIME STEPPING LOOP PLOT FIELDS / OUTPUT / FRAME BUILDER
static uint8_t *TimeSteppingLoopPlotFields_STD(
    int centery, int ie, int je, int n, int plottingInterval,
    double minimumValue, double maximumValue, double scaleValue,
    char filename[], char outputFolder[],
    double **ex, double **ey, double **hz
);

// OMP

// TIME STEPPING LOOP EXEY
static void TimeSteppingLoopUpdateEXEYMain_OMP(
    int ie, int je,
    double **caex, double **cbex, double **caey, double **cbey, double **ex, double **ey, double **hz
);

// TIME STEPPING LOOP HZ
static void TimeSteppingLoopUpdateMagneticFieldHZ_OMP(
    int ie, int is, int je, int js, int n,
    double source[],
    double **dahz, double **dbhz, double **ex, double **ey, double **hz
);

// TIME STEPPING LOOP PLOT FIELDS / OUTPUT / FRAME BUILDER
static uint8_t *TimeSteppingLoopPlotFields_OMP(
    int centery, int ie, int je, int n, int plottingInterval,
    double minimumValue, double maximumValue, double scaleValue,
    char filename[], char outputFolder[],
    double **ex, double **ey, double **hz
);

// OCL

static double *TransformPointerToVector(
    int imax, int jmax,
    double **src
);

static void
TimeSteppingLoopUpdateEXPML_OCL(
    int ie, int iebc, int iefbc, int je, int jebc, int jb, 
    double *caex, double *cbex, 
    double **caexbcb, double **cbexbcb, double **caexbcf, double **cbexbcf,
    double **caexbcl, double **cbexbcl, double **caexbcr, double **cbexbcr,
    double *ex, double **exbcf, double **exbcl, double **exbcr, double **exbcb,
    double *hz, double **hzxbcb, double **hzybcb, double **hzxbcf, double **hzybcf,
    double **hzxbcl, double **hzybcl, double **hzxbcr, double **hzybcr
);

static void
TimeSteppingLoopUpdateEXPMLFront_OCL(
    int ie, int iebc, int iefbc, int jebc, int je, int jb,
    double *caex, double *cbex, double **caexbcf, double **cbexbcf,
    double *ex, double **exbcf, 
    double *hz, double **hzxbcf, double **hzybcf
);

static void
TimeSteppingLoopUpdateEYPML_OCL(
    int ie, int iebc, int iefbc, int je, int jebc,
    double *caey, double **caeybcb, double **cbeybcb, double **caeybcf, double **cbeybcf,
    double **caeybcl, double **cbeybcl, double **caeybcr, double **cbeybcr,
    double *cbey, double *ey, double **eybcb, double **eybcf, double **eybcl, double **eybcr,
    double *hz, double **hzxbcb, double **hzybcb, double **hzxbcf, double **hzybcf,
    double **hzxbcl, double **hzybcl, double **hzxbcr, double **hzybcr
);

static void
TimeSteppingLoopUpdateEYPMLLeft_OCL(
    int iebc, int je,
    double *caey, double **caeybcl, double **cbeybcl,
    double *cbey, double *ey, double **eybcl,
    double *hz, double **hzxbcl, double **hzybcl
);

static void
TimeSteppingLoopUpdateEYPMLRight_OCL(
    int ie, int iebc, int je,
    double *caey, double **caeybcr, double **cbeybcr,
    double *cbey, double *ey, double **eybcr, 
    double *hz, double **hzxbcr, double **hzybcr
);

static void
TimeSteppingLoopUpdateHZXPML_OCL(
    int ie, int iebc, int iefbc, int je, int jebc,
    double **dahzxbcb, double **dbhzxbcb, double **dahzxbcf, double **dbhzxbcf, 
    double **dahzxbcl, double **dbhzxbcl, double **dahzxbcr, double **dbhzxbcr,
    double *ey, double **eybcb, double **eybcf, double **eybcl, double **eybcr,
    double **hzxbcb, double **hzxbcf, double **hzxbcl, double **hzxbcr
);

static void
TimeSteppingLoopUpdateHZXPMLRight_OCL(
    int ie, int je, int iebc,
    double **dahzxbcr, double **dbhzxbcr,
    double *ey, double **eybcr, double **hzxbcr
);

static void
TimeSteppingLoopUpdateHZXPMLLeft_OCL(
    int je, int iebc,
    double **dahzxbcl, double **dbhzxbcl,
    double *ey, double **eybcl, double **hzxbcl
);

static void
TimeSteppingLoopUpdateHZYPML_OCL(
    int ie, int iebc, int iefbc, int je, int jebc,int jb,
    double **dahzybcb, double **dbhzybcb, double **dahzxbcf, double **dbhzxbcf, double **dahzybcf, 
    double **dbhzybcf, double **dahzybcl, double **dbhzybcl, double **dahzybcr, double **dbhzybcr, 
    double *ex, double **exbcb, double **exbcf, double **exbcl, double **exbcr, double **hzybcb,
    double **hzybcf, double **hzybcl, double **hzybcr
);

static void
TimeSteppingLoopUpdateHZYPMLFront_OCL(
    int ie, int iebc, int iefbc, int je, int jebc, int jb,
    double **dahzxbcf, double **dbhzxbcf, double **dahzybcf, double **dbhzybcf,
    double *ex, double **exbcf, double **exbcl, double **exbcr, double **hzybcf
);

static void
TimeSteppingLoopUpdateHZYPMLBack_OCL(
    int ie, int iebc, int iefbc, int je, int jebc, int jb,
    double **dahzybcb, double **dbhzybcb,
    double *ex, double **exbcb, double **exbcl, double **exbcr, double **hzybcb
);

// TIME STEPPING LOOP EXEY
static void TimeSteppingLoopUpdateEXEYMain_OCL(
    int ie, int je, int ib, int jb,
    double *caex, double *cbex, double *caey, double *cbey, double *ex, double *ey, double *hz,
    cl_context context,
    cl_program program,
    cl_command_queue queue,
    cl_mem cl_hz, cl_mem cl_ex, cl_mem cl_ey, cl_mem cl_caex, cl_mem cl_cbex, cl_mem cl_caey, cl_mem cl_cbey
);

static void
TimeSteppingLoopUpdateEXPMLBack_OCL(
    int ie, int iebc, int je, int iefbc, int jebc, int jb,
    double *caex, double *cbex, double **caexbcb, double **cbexbcb,
    double *ex, double **exbcb,
    double *hz, double **hzxbcb, double **hzybcb
);

// TIME STEPPING LOOP HZ
static void TimeSteppingLoopUpdateMagneticFieldHZ_OCL(
    int ie, int is, int je, int js, int n, int ib, int jb,
    double source[],
    double *dahz, double *dbhz, double *ex, double *ey, double *hz,
    cl_context context,
    cl_program program,
    cl_command_queue queue,
    cl_mem cl_hz, cl_mem cl_ex, cl_mem cl_ey, cl_mem cl_dahz, cl_mem cl_dbhz
);

// TIME STEPPING LOOP PLOT FIELDS / OUTPUT / FRAME BUILDER
static uint8_t *TimeSteppingLoopPlotFields_OCL(
    int centery, int ie, int je, int n, int plottingInterval,
    double minimumValue, double maximumValue, double scaleValue,
    char filename[], char outputFolder[],
    double *ex, double *ey, double *hz,
    cl_context context,
    cl_program program,
    cl_command_queue queue,
    cl_mem cl_gif_frame, cl_mem cl_hz
);


// FREES ALL MEMORY FROM POINTERS
void FreeMemoryUsages(
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
);

#endif
