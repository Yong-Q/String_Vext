#include "triclinic_pair.h"

extern "C" double run_triclinic_pair(const double *p) {
    return triclinic_pair_lj(p[0], p[1], p[2], p[3], p[4], p[5],
                             p + 6, p + 9, p + 12,
                             p[15], p[16], p[17], p[18], p[19]);
}
