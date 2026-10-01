#ifndef LEGACY8238_POLICY_H
#define LEGACY8238_POLICY_H
#include <math.h>
#ifdef __CUDACC__
#define LEGACY_HD __host__ __device__
#else
#define LEGACY_HD
#endif
LEGACY_HD inline int legacy_current_flag(double movement, double threshold) {
    return isfinite(movement) && isfinite(threshold) && movement >= 0.
        && threshold > 0. && movement < threshold;
}
struct LegacyLogScore { bool valid; double logd; };
inline double legacy_logadd(double x, double y) {
    if (x == -INFINITY) return y;
    if (y == -INFINITY) return x;
    const double hi = fmax(x,y);
    return hi + log1p(exp(fmin(x,y)-hi));
}
// Unwrapped Cartesian arc, shifted-energy trapezoid: same uncapped TST formula.
inline LegacyLogScore legacy_log_score(const double *a, const double *b,
    const double *c, const double *energy, int n, const double *A,
    const double *B, const double *C, double T, double mass, double hop) {
    LegacyLogScore bad = {false, 0.};
    if (n < 2 || !(T>0) || !(mass>0) || !(hop>0)
        || !isfinite(T) || !isfinite(mass) || !isfinite(hop)) return bad;
    double emin=INFINITY, emax=-INFINITY;
    for (int i=0;i<n;i++) {
        if (!isfinite(a[i]) || !isfinite(b[i]) || !isfinite(c[i])
            || !isfinite(energy[i])) return bad;
        emin=fmin(emin,energy[i]); emax=fmax(emax,energy[i]);
    }
    double log_integral=-INFINITY;
    for (int i=1;i<n;i++) {
        const double da=a[i]-a[i-1], db=b[i]-b[i-1], dc=c[i]-c[i-1];
        const double x=A[0]*da+A[1]*db+A[2]*dc;
        const double y=B[0]*da+B[1]*db+B[2]*dc;
        const double z=C[0]*da+C[1]*db+C[2]*dc;
        const double ds=hypot(hypot(x,y),z)*1e-10;
        if (!isfinite(ds)) return bad;
        if (ds==0) continue;
        const double term=log(ds)-log(2.)+legacy_logadd(
            -(energy[i-1]-emin)/T, -(energy[i]-emin)/T);
        log_integral=legacy_logadd(log_integral,term);
    }
    const double logd=log(.5)+2*log(hop*1e-10)
        +.5*log(1.38e-23*T/(2*3.141592653589793*(mass/1000/6.02214076e23)))
        -(emax-emin)/T-log_integral;
    return {isfinite(logd), logd};
}
inline bool legacy_choose_initial(LegacyLogScore initial, LegacyLogScore final) {
    return initial.valid && (!final.valid || initial.logd > final.logd);
}
#undef LEGACY_HD
#endif
