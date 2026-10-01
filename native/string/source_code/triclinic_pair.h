#ifndef STRING_TRICLINIC_PAIR_H
#define STRING_TRICLINIC_PAIR_H

#include <math.h>

#ifdef __CUDACC__
#define STRING_PBC_HD __host__ __device__
#else
#define STRING_PBC_HD
#endif

// A/B/C are Cartesian rows of the fractional-to-Cartesian matrix.
// Enumerate every lattice image in cutoff, not just one componentwise MIC.
STRING_PBC_HD static inline double triclinic_pair_lj(
    double x, double y, double z, double fa, double fb, double fc,
    const double *A, const double *B, const double *C,
    double epsilon_guest, double sigma_guest,
    double epsilon_frame, double sigma_frame, double cutoff)
{
    const double det = A[0]*(B[1]*C[2]-B[2]*C[1])
                     - A[1]*(B[0]*C[2]-B[2]*C[0])
                     + A[2]*(B[0]*C[1]-B[1]*C[0]);
    if (!isfinite(det) || fabs(det) < 1e-12 || cutoff <= 0.0) return NAN;
    const double ia[3] = {(B[1]*C[2]-B[2]*C[1])/det,
                         (A[2]*C[1]-A[1]*C[2])/det,
                         (A[1]*B[2]-A[2]*B[1])/det};
    const double ib[3] = {(B[2]*C[0]-B[0]*C[2])/det,
                         (A[0]*C[2]-A[2]*C[0])/det,
                         (A[2]*B[0]-A[0]*B[2])/det};
    const double ic[3] = {(B[0]*C[1]-B[1]*C[0])/det,
                         (A[1]*C[0]-A[0]*C[1])/det,
                         (A[0]*B[1]-A[1]*B[0])/det};
    double da = ia[0]*x+ia[1]*y+ia[2]*z-fa;
    double db = ib[0]*x+ib[1]*y+ib[2]*z-fb;
    double dc = ic[0]*x+ic[1]*y+ic[2]*z-fc;
    // This only changes the representative of the complete periodic sum.
    da -= floor(da); db -= floor(db); dc -= floor(dc);
    const double ra = cutoff*sqrt(ia[0]*ia[0]+ia[1]*ia[1]+ia[2]*ia[2]);
    const double rb = cutoff*sqrt(ib[0]*ib[0]+ib[1]*ib[1]+ib[2]*ib[2]);
    const double rc = cutoff*sqrt(ic[0]*ic[0]+ic[1]*ic[1]+ic[2]*ic[2]);
    const double tolerance = 1e-12;
    const int alo = (int)ceil(da-ra-tolerance), ahi = (int)floor(da+ra+tolerance);
    const int blo = (int)ceil(db-rb-tolerance), bhi = (int)floor(db+rb+tolerance);
    const int clo = (int)ceil(dc-rc-tolerance), chi = (int)floor(dc+rc+tolerance);
    const double sigma = (sigma_guest+sigma_frame)*0.5;
    const double epsilon = sqrt(epsilon_guest*epsilon_frame);
    const double sigma2 = sigma*sigma, cutoff2 = cutoff*cutoff;
    const double rcut2 = sigma2/cutoff2;
    const double rcut6 = rcut2*rcut2*rcut2;
    double total = 0.0;
    for (int i=alo; i<=ahi; ++i) {
        for (int j=blo; j<=bhi; ++j) {
            for (int k=clo; k<=chi; ++k) {
                const double u=da-i, v=db-j, w=dc-k;
                const double dx=A[0]*u+A[1]*v+A[2]*w;
                const double dy=B[0]*u+B[1]*v+B[2]*w;
                const double dz=C[0]*u+C[1]*v+C[2]*w;
                const double distance2=fmax(dx*dx+dy*dy+dz*dz, 0.01*sigma2);
                if (distance2 < cutoff2) {
                    const double ratio2=sigma2/distance2;
                    const double ratio6=ratio2*ratio2*ratio2;
                    total += 4.0*epsilon*(ratio6*ratio6-ratio6-rcut6*rcut6+rcut6);
                }
            }
        }
    }
    return total;
}

#undef STRING_PBC_HD
#endif
