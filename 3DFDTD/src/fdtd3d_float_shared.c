///* FD3D_4.2.c. 3D FDTD, plane wave on a dielectric sphere*/
//********************************************************************
//********************************************************************
//
//   3D FDTD code from the book 
//
//   Electromagnetic simulation using the FDTD method - pages 79ff.
//
//   Denis M Sullivan 
//
//   Publisher: IEEE Press, Piscataway New Jersey, 2000 
//   ISBN: 0-7803-4747-1
//
//********************************************************************
//********************************************************************
#include <math.h>	// Functions
#include <stdlib.h>	// Malloc and Free
#include <stdio.h>	// Printf
#include <omp.h>	// OpenMP

#define IE 300
#define JE 300
#define KE 300
#define ia  20
#define ja  20
#define	ka  20

#define NFREQS 3

//
//---- Need to define variables in global space as there
//     is not enough stack space when 3D grid increases 
//

float dx[IE][JE][KE], dy[IE][JE][KE], dz[IE][JE][KE];
float ex[IE][JE][KE], ey[IE][JE][KE], ez[IE][JE][KE];
float hx[IE][JE][KE], hy[IE][JE][KE], hz[IE][JE][KE];
float ix[IE][JE][KE], iy[IE][JE][KE], iz[IE][JE][KE];
float gax[IE][JE][KE], gay[IE][JE][KE], gaz[IE][JE][KE];
float gbx[IE][JE][KE], gby[IE][JE][KE], gbz[IE][JE][KE];
int l, m, n, i, j, k, ic, jc, kc, nsteps, n_pml;
float ddx, dt, T, epsz, muz, pi, eaf, npml;
int ib, jb, kb;
float xn, xxn, xnum, xd, curl_e;
float t0, spread, pulse;
FILE* fp, * fopen();
float ez_inc[JE], hx_inc[JE];
float ez_low_m1, ez_low_m2, ez_high_m1, ez_high_m2;

float idxl[ia][JE][KE], idxh[ia][JE][KE];
float ihxl[ia][JE][KE], ihxh[ia][JE][KE];
float idyl[IE][ja][KE], idyh[IE][ja][KE];
float ihyl[IE][ja][KE], ihyh[IE][ja][KE];
float idzl[IE][JE][ka], idzh[IE][JE][ka];
float ihzl[IE][JE][ka], ihzh[IE][JE][ka];
int ixh, jyh, kzh;

float gi1[IE], gi2[IE], gi3[IE];
float gj1[JE], gj2[JE], gj3[JE];
float gk1[KE], gk2[KE], gk3[KE];
float fi1[IE], fi2[IE], fi3[IE];
float fj1[JE], fj2[JE], fj3[JE];
float fk1[KE], fk2[KE], fk3[KE];

float curl_h, curl_d;

float radius[10], epsilon[10], sigma[10], eps, cond;
int ii, jj, kk, numsph;
float dist, xdist, ydist, zdist;
float freq[NFREQS], arg[NFREQS];
float real_pt[NFREQS][IE][JE], imag_pt[NFREQS][IE][JE];
float amp[IE][JE], phase[IE][JE];
float real_in[5], imag_in[5], amp_in[5], phase_in[5];

/**
 *  Main program
 *
 */
int main(int argc, char **argv)
{
	int zdebug = 0; /* = 1 for ON, = 0 for OFF */
	int xdebug = 1; /*Time Step Control */

        //
        //---- Details of the simulation 
        //
 
        printf("\n\n");
        printf("     FDTD simulation of plane wave scattering \n");
        printf("     ========================================   ");
        printf("\n\n");
        printf("     Number of grid points: \n");
        printf("       x-dimension (IE) = %4d \n", IE);
        printf("       y-dimension (JE) = %4d \n", JE);
        printf("       z-dimension (KE) = %4d \n", KE);
        printf("\n");
        printf("     Number of frequencies (NFREQS) = %3d", NFREQS);
        printf("\n\n");

        //
        //---- It is useful to remember that:
        //
        //        ic is the cell number at the centre of the "i" dimension
        //        jc is the cell number at the centre of the "j" dimension
        //        kc is the cell number at the centre of the "k" dimension
        //

	ic = IE / 2;
	jc = JE / 2;
	kc = KE / 2;

	ib = IE - ia - 1;
	jb = JE - ja - 1;
	kb = KE - ka - 1;

        printf("     Cell locations: ");
        printf("\n\n");
        printf("       (ic,jc,kc) = (%d,%d,%d) \n", ic,jc,kc);
        printf("       (ia,ja,ka) = (%d,%d,%d) \n", ia,ja,ka);
        printf("       (ib,jb,kb) = (%d,%d,%d)   ", ib,jb,kb);
        printf("\n\n");

        //
        //---- Define physical constants 
        //

	pi   = 3.14159;
	epsz = 8.8e-12;
	muz  = 4 * pi * 1.e-7;
	ddx  = .01;	       /* Cell size  in metres  */
	dt   = ddx / 6e8;      /* Time Steps in seconds */

	/* Initialize the arrays */
	for (j = 0; j < JE; j++) {
		ez_inc[j] = 0.;
		hx_inc[j] = 0.;
		for (k = 0; k < KE; k++) {
			for (i = 0; i < IE; i++) {
				ex[i][j][k] = 0.0;
				ey[i][j][k] = 0.0;
				ez[i][j][k] = 0.0;
				dx[i][j][k] = 0.0;
				dy[i][j][k] = 0.0;
				dz[i][j][k] = 0.0;
				hx[i][j][k] = 0.0;
				hy[i][j][k] = 0.0;
				hz[i][j][k] = 0.0;
				ix[i][j][k] = 0.0;
				iy[i][j][k] = 0.0;
				iz[i][j][k] = 0.0;
				gax[i][j][k] = 1.;
				gay[i][j][k] = 1.;
				gaz[i][j][k] = 1.;
				gbx[i][j][k] = 0.;
				gby[i][j][k] = 0.;
				gbz[i][j][k] = 0.;
			}
		}
	}

	for (n = 0; i < NFREQS; n++) {
		real_in[n] = 0.;
		imag_in[n] = 0.;
		for (j = 0; j < JE; j++) {
			for (i = 0; i < IE; i++) {
				real_pt[n][i][j] = 0.;
				imag_pt[n][i][j] = 0.;
			}
		}
	}


	/* Parameters for the Fourier Transforms */

	freq[0] = 10.e6;
	freq[1] = 100.e6;
	freq[2] = 433.e6;

        printf("     Parameters for each frequency - Fourier transform");
        printf("\n\n");

	for(n = 0; n < NFREQS; n++) 
           {
	    arg[n] = 2 * pi * freq[n] * dt;
	
    	    /* printf("    %2d  %6.2f  %7.5f \n", n, freq[n] * 1.e-6, arg[n]);	Commented Out for Neater TimeStep */
	   }
 
        printf("\n\n");

	for (i = 0; i < ia; i++) {
		for (j = 0; j < JE; j++) {
			for (k = 0; k < KE; k++) {
				idxl[i][j][k] = 0.0;
				idxh[i][j][k] = 0.0;
				ihxl[i][j][k] = 0.0;
				ihxh[i][j][k] = 0.0;
			}
		}
	}

	for (i = 0; i < IE; i++) {
		for (j = 0; j < ja; j++) {
			for (k = 0; k < KE; k++) {
				idyl[i][j][k] = 0.0;
				idyh[i][j][k] = 0.0;
				ihyl[i][j][k] = 0.0;
				ihyh[i][j][k] = 0.0;
			}
		}
	}

	for (i = 0; i < IE; i++) {
		for (j = 0; j < JE; j++) {
			for (k = 0; k < ka; k++) {
				idzl[i][j][k] = 0.0;
				idzh[i][j][k] = 0.0;
				ihzl[i][j][k] = 0.0;
				ihzh[i][j][k] = 0.0;
			}
		}
	}

	/*	Boundary Conditions */

	for (i = 0; i < IE; i++) {
		gi1[i] = 0.;
		fi1[i] = 0.;
		gi2[i] = 1.;
		fi2[i] = 1.;
		gi3[i] = 1.;
		fi3[i] = 1.;
		}

	for (j = 0; j < JE; j++) {
		gj1[j] = 0.;
		fj1[j] = 0.;
		gj2[j] = 1.;
		fj2[j] = 1.;
		gj3[j] = 1.;
		fj3[j] = 1.;
	}

	for (k = 0; k < IE; k++) {
		gk1[k] = 0.;
		fk1[k] = 0.;
		gk2[k] = 1.;
		fk2[k] = 1.;
		gk3[k] = 1.;
		fk3[k] = 1.;
	}

        //
        //---- Read the number of PML 
        //

	printf("Enter the value for number of points in PML (npml) --> ");
	(void) scanf("%f", &npml);
	printf("Value read for npml = %f \n", npml);

	n_pml = npml;

        //

        printf("     Computed values for each PML ");
        printf("\n\n");

	for(i = 0; i < n_pml; i++) 
           {
	    xxn = (npml - i) / npml;
	    xn = .33 * pow(xxn, 3.);

	   /*  printf("    %d   xn = %8.4f   xn = %8.4f \n", i, xxn, xn);	Commented Out for Tidy TimeStep	 */

		fi1[i] = xn;
		fi1[IE - i - 1] = xn;
		gi2[i] = 1. / (1.+xn);
		gi2[IE - i - 1] = 1. / (1.+xn);
		gi3[i] = (1.-xn) / (1.+xn);
		gi3[IE - i - 1] = (1.-xn) / (1.+xn);

		xxn = (npml-i-.5)/npml;
		xn = .33*pow(xxn, 3.);

		gi1[i] = xn;
		gi1[IE - i - 2] = xn;
		fi2[i] = 1. / (1.+xn);
		fi2[IE - i - 2] = 1. / (1.+xn);
		fi3[i] = (1.-xn) / (1.+xn);
		fi3[IE - i - 2] = (1.-xn) / (1.+xn);
	}

      printf("\n\n");
	printf("     Contents of the fi1/gi2/gi3 matrices  \n");
        printf("\n\n");

	if(zdebug)
	{
	for (i = 0; i < IE; i++)
	 {
		printf("%2d %6.4f %6.4f %6.4f\n",
			i, fi1[i], gi2[i], gi3[i]);

		printf("	%6.4f %6.4f %6.4f\n",
				gi1[i], fi2[i], fi3[i]);
	}
	 // End for (i...)
	}
	// End of if (zdebug)
	

	for (j = 0; j < n_pml; j++) {
		xxn = (npml-j) / npml;
		xn = .33*pow(xxn, 3.);

		fj1[j] = xn;
		fj1[JE - j - 1] = xn;
		gj2[j] = 1. / (1.+xn);
		gj2[JE - j - 1] = 1. / (1.+xn);
		gj3[j] = (1.-xn) / (1.+xn);
		gj3[JE - j - 1] = (1.-xn) / (1.+xn);

		xxn = (npml-j-.5) / npml;
		xn = .33*pow(xxn, 3.);

		gj1[j] = xn;
		gj1[JE - j - 2] = xn;
		fj2[j] = 1. / (1.+xn);
		fj2[JE - j - 2] = 1. / (1.+xn);
		fj3[j] = (1.-xn) / (1.+xn);
		fj3[JE - j - 2] = (1.-xn) / (1.+xn);
	}

        //

	printf("     Contents of the fj & gj vectors");
        printf("\n\n");

	if (zdebug)
	{
	for (j = 0; j < JE; j++) 
		{
		printf("%2d %6.4f %6.4f %6.4f\n",
			j, fj1[j], gj2[j], gj3[j]);

		printf("	%6.4f %6.4f %6.4f\n",
			gj1[j], fj2[j], fj3[j]);
		}
		//End of (j...)
	}
	//End of if(zdebug)

	for (k = 0; k < n_pml; k++) {
		xxn = (npml - k) / npml;
		xn = .33 * pow(xxn, 3.);

		fk1[k] = xn;
		fk1[KE - k - 1] = xn;
		gk2[k] = 1. / (1.+xn);
		gk2[KE - k - 1] = 1. / (1.+xn);
		gk3[k] = (1.-xn) / (1.+xn);
		gk3[KE - k - 1] = (1.-xn) / (1.+xn);

		xxn = (npml - k - .5) / npml;
		xn = .33 * pow(xxn, 3.);

		gk1[k] = xn;
		gk1[KE - k - 2] = xn;
		fk2[k] = 1. / (1.+xn);
		fk2[KE - k - 2] = 1. / (1.+xn);
		fk3[k] = (1.-xn) / (1.+xn);
		fk3[KE - k - 2] = (1.-xn) / (1.+xn);
	}

	printf("     Contents of the fk & gk vectors");
        printf("\n\n");

	if(zdebug)
	{
	for (k = 0; k < KE; k++) 
		{
		printf("%2d %6.4f %6.4f %6.4f\n",
			k, fk1[k], gk2[k], gk3[k]);
	
		printf("	%6.4f %6.4f %6.4f\n",
			gk1[k], fk2[k], fk3[k]);
		}
		//End for (k...)
	}
	// End of if(zdebug)

        //
	//---- Specify the dielectric sphere(s)
        //

	epsilon[0] = 1.;
	sigma[0]   = 0;

	printf("Enter the number of dielectric spheres --> ");

	(void) scanf("%d", &numsph);

	printf("Number of dielectric spheres (numsph) = %d \n ", numsph);

        //
        //---- For each spehere, enter the radius in "cells" 
        //     plus epsilon and sigma for the material
        //

	for(n = 1; n <= numsph; n++) 
           {
	    printf("Enter sphere radius (cells), epsilon, sigma --> ");
		
            (void) scanf("%f %f %f", &radius[n], &epsilon[n], &sigma[n]);
		
            printf("   Values read - Radius = %6.2f	Eps = %6.2f	Sigma = %6.2f \n ",
			radius[n], epsilon[n], sigma[n]);
	   }

        //
        //---- Summary of all materials 
        //

        printf("\n\n");
        printf("     Summary of %d dielectrics: ",numsph+1);
        printf("\n\n");
        printf("     Note: entry [0] is for free space and has radius \"0\" ");
        printf("\n\n");
        printf("           Other dieletrics are arranged in concentric circles \n");
        printf("           around the centre of the grid - (ic,jc,kc) = (%d,%d,%d)",ic,jc,kc);
        printf("\n\n");
        printf("     Index   Radius   Epsilon   Sigma  \n");
        printf("     -----   ------   -------  ------- \n");

	if (zdebug)
	{
	for(n = 0; n <= numsph; n++) 
           {
	    printf("    %6d   %6.2f   %7.2f  %6.2f\n", n,radius[n],epsilon[n],sigma[n]);
	   }
	   //End of (n...)
	}
	// End of if(zdebug)

        printf("\n\n");
        printf("     **** End of details of dielectric spheres ");
        printf("\n\n");

        /*                     */
	/*  Calculate gax,gbx  */
        /*                     */
 
	for(i = ia; i < ib; i++) 
           {
	    for(j = ja; j < jb; j++) 
               {
		for(k = ka; k < kb; k++) 
                   {
                    //
                    //.... Calculate distance from centre 
                    //     of the grid (ic,jc,kc)
                    //

		    xdist = (ic - i - .5);
		    ydist = (jc - j);
		    zdist = (kc - k);
				
                    dist  = sqrt(pow(xdist, 2.) + pow(ydist, 2.) + pow(zdist, 2.));
                                 
                    //
                    //.... Based on distance from centre 
                    //     of the grid, select the value of
                    //     epsilon and sigma for this cell (i,j,k)
                    //
                    //     Note that the default is such that 
                    //     type "0" is free space. If the current
                    //     cell lies outside the radius of any ring
                    //     of dielectric then it defaults to free 
                    //     space values
                    //

		    eps  = epsilon[0];
		    cond = sigma[0];

	            for(n = 1; n <= numsph; n++) 
                       {
		        if(dist <= radius[n]) 
                          {
		           eps  = epsilon[n];
			   cond = sigma[n];
		          }
		       }

                    //
                    //.... Compute gax and gbx for the cell
                    //

		    gax[i][j][k] = 1. /(eps+(cond*dt/epsz));
		    gbx[i][j][k] = cond*dt/epsz;
		   }
                    // End loop over k axis 
	       }
                // End loop over j axis
	   }
            // End loop over the i axis

	printf("     Contents of the the Gax 3D matrix");
        printf("\n\n");

	if (zdebug)
	{
	for (j = ja; j <= jb; j++) 
		{
		printf("%3d", j);

		for (i = ia; i <= ib; i++)
			 {
			printf("%5.2f", gax[i][j][kc]);
		}
		printf(" \n");
	}
	// End for (j...)
	}
	// End of if(zdebug)	

        printf("\n\n");
	printf("     End of the Gax 3D matrix");
        printf("\n\n");
   
        //fclose(fp);

	/*		Calculate gay,gby	*/
	
	for (i = ia; i < ib; i++) {
		for (j = ja; j < jb; j++) {
			for (k = ka; k < kb; k++) {
				eps  = epsilon[0];
				cond = sigma[0];

				xdist = (ic - i);
				ydist = (jc-j-.5);
				zdist = (kc - k);
				dist = sqrt(pow(xdist, 2.) + pow(ydist, 2.) + pow(zdist, 2.));
				for (n = 1; n <= numsph; n++) {
					if (dist <= radius[n]) {
						eps = epsilon[n];
						cond = sigma[n];
					}
				}

				gay[i][j][k] = 1. / (eps+(cond*dt/epsz));
				gby[i][j][k] = cond*dt/epsz;
			}
		}
	}

        //

	printf("    Contents of the 3D Gay matrix ");
        printf("\n\n");
	
	if(zdebug)
	{
	for(j = ja; j <= jb; j++) 
           {
		printf("%3d", j);

		for (i = ia; i <= ib; i++) 
		{
			printf("%5.2f", gay[i][j][kc]);
		}
		printf(" \n");
	   }
		//End for (j...)
	}
	// End of if(zdebug)

        //fclose(fp);

	/*		Calculate gaz,gbz	*/

	for (i = ia; i < ib; i++) {
		for (j = ja; j < jb; j++) {
			for (k = ka; k < kb; k++) {
				eps = epsilon[0];
				cond = sigma[0];

				xdist = (ic - i);
				ydist = (jc - j);
				zdist = (kc-k-.5);
				dist = sqrt(pow(xdist, 2.) + pow(ydist, 2.) + pow(zdist, 2.));
				for (n = 1; n <= numsph; n++) {
					if (dist <= radius[n]) {
						eps = epsilon[n];
						cond = sigma[n];
					}
				}

				gaz[i][j][k] = 1. /(eps+(cond*dt/epsz));
				gbz[i][j][k] = cond*dt/epsz;
			}
		}
	}

	if(zdebug)
	{
	printf(" Gaz	\n");
	for (j = ja; j <= jb; j++)
		 {
		printf("%3d", j);

		for (i = ia; i <= ib; i++)
			 {
			printf("%5.2f", gaz[i][j][kc]);
		}
		printf(" \n");
	}
	// End for (j...)
	}
	// End of if(zdebug)	

        //fclose(fp);

        //=================================================================
        //
        //   T I M E    L O O P 
        //
        //=================================================================

	if(xdebug)
	{
        printf("\n\n");
        printf("     Entering time loop ... ");
        printf("\n\n");
	}
	//End of xdebug
	
        nsteps = 150;
	
	if(xdebug)
	{
	printf("     Number of time steps = %d", nsteps);
        printf("\n\n");
	}
	//End of xdebug
	
	t0     = 40.0;
	spread = 10.0;
	T      = 0;

	//================================================================
	//
	//   P R E P A R E   T H R E A D I N G   E N V I R O N M E N T
	//
	//================================================================

	int const nMaxOMPThreads = omp_get_max_threads();
	int const nNumProcessors = omp_get_num_procs();

	printf("     OMP system reports following parameters: \n");
	printf("         max number of threads    = %6d \n", nMaxOMPThreads);
	printf("         number of computer nodes = %6d   ", nNumProcessors);
	printf("\n\n");	


	
        for(n = 1; n <= nsteps; n++) 
           {
	    T = T + 1;

            printf("\n\n "); 
              for(int icol=2; icol<72; ++icol) printf("-");
     	    printf("\n\n");
     	    printf("     Timestep (n) = %d of %d ", n, nsteps);
            printf("\n\n");
		 

		/*	----	Start of the Main FDTD Loop	----	*/

			/*	Calculate the Incident Buffer	*/

			for (j = 1; j < JE; j++) {
				ez_inc[j] = ez_inc[j] + .5*(hx_inc[j - 1] - hx_inc[j]);
			}

		/*	Fourier Transform of the Incident Field	*/
			for (m = 0; m < NFREQS; m++)
			{
				real_in[m] = real_in[m] + cos(arg[m]*T)*ez_inc[ja - 1];
				imag_in[m] = imag_in[m] - sin(arg[m]*T)*ez_inc[ja - 1];
			}

			/*	Source	*/

		/*	pulse = sin(2*pi*400*1e6*dt*T);	*/
			pulse = exp(-.5*(pow((t0-T)/spread, 2.0) ));
			ez_inc[3] = pulse;
			printf("     Time (T) = %4.0f. Magnitude of pulse = %6.2f \n ", T, pulse);
                        printf("\n\n");

		/*	Boundary conditions for the Incident Buffer	*/

			ez_inc[0] = ez_low_m2;
			ez_low_m2 = ez_low_m1;
			ez_low_m1 = ez_inc[1];
			ez_inc[JE - 1] = ez_high_m2;
			ez_high_m2 = ez_high_m1;
			ez_high_m1 = ez_inc[JE - 2];

			/*	Calculate the DX field	*/
			
			#pragma omp parallel for shared(hz, hy) private (i,j,k) schedule(static)
			for (i = 1; i < ia; i++) {
				for (j = 1; j < JE; j++) {
					for (k = 1; k < KE; k++) {
						curl_h = (hz[i][j][k] - hz[i][j - 1][k]
							- hy[i][j][k] + hy[i][j][k - 1]);
						idxl[i][j][k] = idxl[i][j][k] + curl_h;
						dx[i][j][k] = gj3[j] * gk3[k] * dx[i][j][k]
					+ gj2[j]*gk2[k]*.5*(curl_h + gi1[i]*idxl[i][j][k]);
					}
				}
			}

			#pragma omp parallel for shared(hz, hy) private (i,j,k) schedule(static)
			for (i = ia; i <= ib; i++) {
				for (j = 1; j < JE; j++) {
					for (k = 1; k < KE; k++) {
						curl_h = (hz[i][j][k] - hz[i][j - 1][k]
							- hy[i][j][k] + hy[i][j][k - 1]);
						dx[i][j][k] = gj3[j] * gk3[k] * dx[i][j][k]
							+ gj2[j]*gk2[j]*.5*curl_h;
					}
				}
			}

			#pragma omp parallel for shared(ib, hz, hy) private (i,j,k,curl_h) schedule(static)
			for (i = ib + 1; i < IE; i++) {
				ixh = i - ib - 1;
				for (j = 1; j < JE; j++) {
					for (k = 1; k < KE; k++) {
						curl_h = (hz[i][j][k] - hz[i][j - 1][k]
							- hy[i][j][k] + hy[i][j][k - 1]);
						idxh[ixh][j][k] = idxh[ixh][j][k] + curl_h;
						dx[i][j][k] = gj3[j]*gk3[k]*dx[i][j][k]
					+ gj2[j]*gk2[k]*.5*(curl_h + gi1[i]*idxh[ixh][j][k]);
					}
				}
			}

			/*	Calculate the DY Field	*/

			#pragma omp parallel for shared(hx, hz) private (i,j,k) schedule(static)
			for (i = 1; i < IE; i++) {
				for (j = 1; j < ja; j++) {
					for (k = 1; k < KE; k++) {
						curl_h = (hx[i][j][k] - hx[i][j][k - 1]
							- hz[i][j][k] + hz[i - 1][j][k]);
						idyl[i][j][k] = idyl[i][j][k] + curl_h;
						dy[i][j][k] = gi3[i]*gk3[k]*dy[i][j][k]
					+ gi2[i]*gk2[k]*.5*(curl_h+gj1[j]*idyl[i][j][k]);
					}
				}
			}

			#pragma omp parallel for shared(jb, hx, hz) private (i,j,k) schedule(static)
			for (i = 1; i < IE; i++) {
				for (j = ja; j <= jb; j++) {
					for (k = 1; k < KE; k++) {
						curl_h = (hx[i][j][k] - hx[i][j][k - 1]
							- hz[i][j][k] + hz[i - 1][j][k]);
						dy[i][j][k] = gi3[i] * gk3[k] * dy[i][j][k]
							+ gi2[i] * gk2[k] * .5 * curl_h;
					}
				}
			}
			#pragma omp parallel for shared(hx, hz) private (i,j,k) schedule(static)
			for (i = 1; i < IE; i++) {
				for (j = jb + 1; j < JE; j++) {
					jyh = j - jb - 1;
					for (k = 1; k < KE; k++) {
						curl_h = (hx[i][j][k] - hx[i][j][k - 1]
							- hz[i][j][k] + hz[i - 1][j][k]);
						idyh[i][jyh][k] = idyh[i][jyh][k] + curl_h;
						dy[i][j][k] = gi3[i] * gk3[k] * dy[i][j][k]
							+ gi2[i] * gk2[k] * .5 * (curl_h + gj1[j] * idyh[i][jyh][k]);
					}
				}
			}

			/*	Incident Dy	*/
			#pragma omp parallel for shared(ib,jb) private (i,j) schedule(static)
			for (i = ia; i <= ib; i++) {
				for (j = ja; j <= jb - 1; j++) {
					dy[i][j][ka] = dy[i][j][ka] - .5 * hx_inc[j];
					dy[i][j][kb + 1] = dy[i][j][kb + 1] + .5 * hx_inc[j];
				}
			}

			/*	Calculate the DZ field	*/
			#pragma omp parallel for shared(hx, hy) private (i,j,k) schedule(static)
			for (i = 1; i < IE; i++) {
				for (j = 1; j < JE; j++) {
					for (k = 0; k < ka; k++) {
						curl_h = (hy[i][j][k] - hy[i - 1][j][k]
							- hx[i][j][k] + hx[i][j - 1][k]);
						idzl[i][j][k] = idzl[i][j][k] + curl_h;
						dz[i][j][k] = gi3[i] * gj3[j] * dz[i][j][k]
							+ gi2[i] * gj2[j] * .5 * (curl_h + gk1[k] * idzl[i][j][k]);
					}
				}
			}
			#pragma omp parallel for shared(kb, hx, hz) private (i,j,k) schedule(static)
			for (i = 1; i < IE; i++) {
				for (j = 1; j < JE; j++) {
					for (k = ka; k <= kb; k++) {
						curl_h = (hy[i][j][k] - hy[i - 1][j][k]
							- hx[i][j][k] + hx[i][j - 1][k]);
						dz[i][j][k] = gi3[i] * gj3[j] * dz[i][j][k]
							+ gi2[i] * gj2[j] * .5 * curl_h;
					}
				}
			}
			#pragma omp parallel for shared(kb, hx, hz) private (i,j,k) schedule(static)
			for (i = 1; i < IE; i++) {
				for (j = 1; j < JE; j++) {
					for (k = kb + 1; k < KE; k++) {
						kzh = k - kb - 1;
						curl_h = (hy[i][j][k] - hy[i - 1][j][k]
							- hx[i][j][k] + hx[i][j - 1][k]);
						idzh[i][j][kzh] = idzh[i][j][kzh] + curl_h;
						dz[i][j][k] = gi3[i] * gj3[j] * dz[i][j][k]
							+ gi2[i] * gj2[j] * .5 * (curl_h + gk1[k] * idzh[i][j][kzh]);
					}
				}
			}

			/*	Incident DZ	*/
			#pragma omp parallel for shared(ib,kb) private (i,k) schedule(static)
			for (i = ia; i <= ib; i++) {
				for (k = ka; k <= kb; k++) {
					dz[i][ja][k] = dz[i][ja][k] + .5 * hx_inc[ja - 1];
					dz[i][jb][k] = dz[i][jb][k] - .5 * hx_inc[jb];
				}
			}

			/*	Source	*/

		/*		pulse = sin(2*pi*400*1e6*dt*T)
				for (k = kc-6; k <= kc+6; k++) {
					dz[ic][jc][k] = 0 ;
				}
				pulse = exp(-.5*(pow((t0-T)/spread,2.0) ));
				dz[ic][jc][kc] = pulse;

				printf("%4.0f	%6.2f	\n ",T,pulse);	*/

			/*	Calculate the E from D field */
			/*	Remember: Part of the PML is E=0 at the edges */
			#pragma omp parallel for private (i,j,k) schedule(static)
			for (i = 1; i < IE - 1; i++) {
				for (j = 1; j < JE - 1; j++) {
					for (k = 1; k < KE - 1; k++) {
					ex[i][j][k] = gax[i][j][k] * (dx[i][j][k] - ix[i][j][k]);
						ix[i][j][k] = ix[i][j][k] + gbx[i][j][k] * ex[i][j][k];
					ey[i][j][k] = gay[i][j][k] * (dy[i][j][k] - iy[i][j][k]);
						iy[i][j][k] = iy[i][j][k] + gby[i][j][k] * ey[i][j][k];
					ez[i][j][k] = gaz[i][j][k] * (dz[i][j][k] - iz[i][j][k]);
						iz[i][j][k] = iz[i][j][k] + gbz[i][j][k] * ez[i][j][k];
					}
				}
			}

			/*	Calculate the Fourier Transform of Ex	*/
			#pragma omp parallel for private (i,j,m) schedule(static)
			for (j = 0; j < JE; j++) {
				for (i = 0; i < JE; i++) {
					for (m = 0; m < NFREQS; m++) {
						real_pt[m][i][j] = real_pt[m][i][j]
							+ cos(arg[m] * T) * ez[i][j][kc];
						imag_pt[m][i][j] = imag_pt[m][i][j]
							+ sin(arg[m] * T) * ez[i][j][kc];
					}
				}
			}

			/*	Calculate the Incident Field	*/
			for (j = 0; j < JE - 1; j++) {
				hx_inc[j] = hx_inc[j] + .5 * (ez_inc[j] - ez_inc[j + 1]);
			}

			/*	Calculate the Hx Field	*/
			#pragma omp parallel for shared(ey, ez) private (i,j,k) schedule(static)
			for (i = 0; i < ia; i++) {
				for (j = 0; j < JE - 1; j++) {
					for (k = 0; k < KE - 1; k++) {
						curl_e = (ey[i][j][k + 1] - ey[i][j][k]
							- ez[i][j + 1][k] + ez[i][j][k]);
						ihxl[i][j][k] = ihxl[i][j][k] + curl_e;
						hx[i][j][k] = fj3[j] * fk3[k] * hx[i][j][k]
							+ fj2[j] * fk2[k] * .5 * (curl_e + fi1[i] * ihxl[i][j][k]);
					}
				}
			}
			#pragma omp parallel for shared(ib, ey, ez) private (i,j,k) schedule(static)
			for (i = ia; i <= ib; i++) {
				for (j = 0; j < JE - 1; j++) {
					for (k = 0; k < KE - 1; k++) {
						curl_e = (ey[i][j][k + 1] - ey[i][j][k]
							- ez[i][j + 1][k] + ez[i][j][k]);
						hx[i][j][k] = fj3[j] * fk3[k] * hx[i][j][k]
							+ fj2[j] * fk2[k] * .5 * curl_e;
					}
				}
			}
			#pragma omp parallel for shared(ib, ey, ez) private (i,j,k) schedule(static)
			for (i = ib + 1; i < IE; i++) {
				ixh = i - ib - 1;
				for (j = 0; j < JE - 1; j++) {
					for (k = 0; k < KE - 1; k++) {
						curl_e = (ey[i][j][k + 1] - ey[i][j][k]
							- ez[i][j + 1][k] + ez[i][j][k]);
						ihxh[ixh][j][k] = ihxh[ixh][j][k] + curl_e;
						hx[i][j][k] = fj3[j] * fk3[k] * hx[i][j][k]
							+ fj2[j] * fk2[k] * .5 * (curl_e + fi1[i] * ihxh[ixh][j][k]);
					}
				}
			}

			/*	Calculate the Hy Field	*/
			#pragma omp parallel for shared(ex, ez) private (i,j,k) schedule(static)
			for (i = 0; i < IE - 1; i++) {
				for (j = 0; j < ja; j++) {
					for (k = 0; k < KE - 1; k++) {
						curl_e = (ez[i + 1][j][k] - ez[i][j][k]
							- ex[i][j][k + 1] + ex[i][j][k]);
						ihyl[i][j][k] = ihyl[i][j][k] + curl_e;
						hy[i][j][k] = fi3[i] * fk3[k] * hy[i][j][k]
							+ fi2[i] * fk3[k] * .5 * (curl_e + fj1[j] * ihyl[i][j][k]);
					}
				}
			}
			#pragma omp parallel for shared(jb, ex, ez) private (i,j,k) schedule(static)
			for (i = 0; i < IE - 1; i++) {
				for (j = ja; j <= jb; j++) {
					for (k = 0; k < KE - 1; k++) {
						curl_e = (ez[i + 1][j][k] - ez[i][j][k]
							- ex[i][j][k + 1] + ex[i][j][k]);
						hy[i][j][k] = fi3[i] * fk3[k] * hy[i][j][k]
							+ fi2[i] * fk3[k] * .5 * curl_e;
					}
				}
			}
			#pragma omp parallel for shared(jb, ex, ez) private (i,j,k) schedule(static)
			for (i = 0; i < IE - 1; i++) {
				for (j = jb + 1; j < JE; j++) {
					jyh = j - jb - 1;
					for (k = 0; k < KE - 1; k++) {
						curl_e = (ez[i + 1][j][k] - ez[i][j][k]
							- ex[i][j][k + 1] + ex[i][j][k]);
						ihyh[i][jyh][k] = ihyh[i][jyh][k] + curl_e;
						hy[i][j][k] = fi3[i] * fk3[k] * hy[i][j][k]
							+ fi2[i] * fk3[k] * .5 * (curl_e + fj1[j] * ihyh[i][jyh][k]);
					}
				}
			}

			/*	Incident Hy	*/

			for (j = ja; j <= jb; j++) {
				for (k = ka; k <= kb; k++) {
					hy[ia - 1][j][k] = hy[ia - 1][j][k] - .5 * ez_inc[j];
					hy[ib][j][k] = hy[ib][j][k] + .5 * ez_inc[j];
				}
			}

			/*	Calculate the Hz Field	*/
			#pragma omp parallel for shared(ex, ey) private (i,j,k) schedule(static)
			for (i = 0; i < IE - 1; i++) {
				for (j = 0; j < JE - 1; j++) {
					for (k = 0; k < ka; k++) {
						curl_e = (ex[i][j + 1][k] - ex[i][j][k]
							- ey[i + 1][j][k] + ey[i][j][k]);
						ihzl[i][j][k] = ihzl[i][j][k] + curl_e;
						hz[i][j][k] = fi3[i] * fj3[j] * hz[i][j][k]
							+ fi2[i] * fj2[j] * .5 * (curl_e + fk1[k] * ihzl[i][j][k]);
					}
				}
			}
			#pragma omp parallel for shared(kb, ex, ey) private (i,j,k) schedule(static)
			for (i = 0; i < IE - 1; i++) {
				for (j = 0; j < JE - 1; j++) {
					for (k = ka; k <= kb; k++) {
						curl_e = (ex[i][j + 1][k] - ex[i][j][k]
							- ey[i + 1][j][k] + ey[i][j][k]);
						hz[i][j][k] = fi3[i] * fj3[j] * hz[i][j][k]
							+ fi2[i] * fj2[j] * .5 * curl_e;
					}
				}
			}
			#pragma omp parallel for shared(kb, ex, ey) private (i,j,k) schedule(static)
			for (i = 0; i < IE - 1; i++) {
				for (j = 0; j < JE - 1; j++) {
					for (k = kb + 1; k < KE; k++) {
						kzh = k - kb - 1;
						curl_e = (ex[i][j + 1][k] - ex[i][j][k]
							- ey[i + 1][j][k] + ey[i][j][k]);
						ihzh[i][j][kzh] = ihzh[i][j][kzh] + curl_e;
						hz[i][j][k] = fi3[i] * fj3[j] * hz[i][j][k]
							+ fi2[i] * fj2[j] * .5 * (curl_e + fk1[k] * ihzh[i][j][kzh]);
					}
				}
			}
		}
		 /*	----	End of the Main FDTD Loop	----	*/

                printf("\n\n");
                printf("     Loop over timeateps is completed");
                printf("\n\n");
		printf("     Details of the JC Plane now follow");
                printf("\n\n");
		
		if (zdebug)
		{
		printf("Ez \n");
		for (k = 0; k < KE; k++)
			 {
			printf("%2d	", k);

			for (i = 0; i < IE; i++)
			 	{
				printf("%6.3f", ez[i][jc][k]);
				}
			printf(" \n");
			}	
			// End for(k...)
		}
		 // End of if(zdebug)

		/*	Write the E field out to a File "Ez" */
		fp = fopen("Ez", "w");
		for (j = 0; j < JE; j++) {
			for (i = 0; i < IE; i++) {
				fprintf(fp, "%9.6f ", ez[i][j][kc]);
			}
			fprintf(fp, " \n");
		}
		fclose(fp);

		/*	Write the E field out to a file "Ezk" */
		fp = fopen("Ezk", "w");
		for (k = 0; k < KE; k++) {
			for (i = 0; i < IE; i++) {
				fprintf(fp, "%7.4f ", ez[i][jc][k]);
			}
			fprintf(fp, " \n");
		}
		fclose(fp);

		printf("T = %4.0f \n", T);

	/*	Calculate the Fouier amplitude and phase of the incident pulse */
	/*	for (m = 0; m < NFREQS; m++) {
			amp_in[m] = sqrt(pow(real_in[m], 2.) + pow(imag_in[m], 2.));
			phase_in[m] = atan2(imag_in[m], real_in[m]);
			printf("%d		Input Pulse:	%8.4f	%8.4f	%8.4f	%7.2f\n",
				m, real_in[m], imag_in[m], amp_in[m], (180.0 / pi)* phase_in[m]);
		} */
	
	/*	Calculate the Fouier amplitude and phase of the Total field field */
	/*	for (m = 0; m < NFREQS; m++) {
			if (m == 0)			fp = fopen("amp1", "w");
			else if (m == 1)	fp = fopen("amp2", "w");
			else if (m == 2)	fp = fopen("amp3", "w");
			{
				printf("%2d	%7.2f	MHz\n", m, freq[m] * 1.e-6);
				for (j = ja; j <= jb; j++) {
					if (gaz[ic][j][kc] < 1.00) {
						amp[ic][j] = (1. / amp_in[m]
							* sqrt(pow(real_pt[m][ic][j], 2.) + pow(imag_pt[m][ic][j], 2.)));
						printf("%2d	%9.4f	\n", jc - j, amp[ic][j]);
						fprintf(fp, "	%9.4f	\n", amp[ic][j]);
					}
				}
			}
			fclose(fp);
		}	*/
//	}
   //
   //---- End of the main program 
   //

   printf("\n\n");
   printf("     **** End of program");
   printf("\n\n");

   return(1);
  }

//********************************************************************
//********************************************************************
//
//   End of file 
//
//********************************************************************
//********************************************************************

