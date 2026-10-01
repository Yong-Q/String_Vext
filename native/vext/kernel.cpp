using int64_t = long long;
using int32_t = int;
static_assert(sizeof(int64_t) == 8 && sizeof(int32_t) == 4);

extern "C" int evaluate_reused_neighbors(
    int n_centers, int n_axes, int n_images, double cutoff2,
    const double *centers, const double *axes, const double *images,
    const double *offsets, const double *coeff12, const double *coeff6,
    const double *shift, const double *floor2,
    const int64_t *starts, const int32_t *indices, double *output) {
    if (n_centers < 0 || n_axes < 1 || n_images < 1 || cutoff2 <= 0)
        return 1;
    for (int center = 0; center < n_centers; ++center) {
        if (starts[center] > starts[center + 1])
            return 2;
        for (int axis = 0; axis < n_axes; ++axis) {
            double energy = 0.0;
            for (int site = 0; site < 2; ++site) {
                const double x = centers[3 * center] + offsets[site] * axes[3 * axis];
                const double y = centers[3 * center + 1] + offsets[site] * axes[3 * axis + 1];
                const double z = centers[3 * center + 2] + offsets[site] * axes[3 * axis + 2];
                for (int64_t neighbor = starts[center]; neighbor < starts[center + 1]; ++neighbor) {
                    const int32_t index = indices[neighbor];
                    if (index < 0 || index >= n_images)
                        return 3;
                    const double dx = x - images[3 * index];
                    const double dy = y - images[3 * index + 1];
                    const double dz = z - images[3 * index + 2];
                    const double distance2 = dx * dx + dy * dy + dz * dz;
                    if (distance2 >= cutoff2)
                        continue;
                    const double safe2 = distance2 < floor2[index] ? floor2[index] : distance2;
                    const double inverse2 = 1.0 / safe2;
                    const double inverse6 = inverse2 * inverse2 * inverse2;
                    energy += coeff12[index] * inverse6 * inverse6
                              - coeff6[index] * inverse6 + shift[index];
                }
            }
            if (!__builtin_isfinite(energy))
                return 4;
            output[center * n_axes + axis] = energy;
        }
    }
    return 0;
}
