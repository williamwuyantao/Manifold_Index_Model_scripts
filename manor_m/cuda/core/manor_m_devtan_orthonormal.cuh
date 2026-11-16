#pragma once
// DevTan v2. Restore the evaluator's orthonormal-basis
// invariant in FP64 without changing the stored/trained tangent or true input.
__device__ __forceinline__ double manor_devtan_warp_sum(double value) {
    for (int offset=16;offset>0;offset>>=1)
        value+=__shfl_down_sync(0xffffffffu,value,offset);
    return __shfl_sync(0xffffffffu,value,0);
}

__device__ __forceinline__ float manor_devtan_orthonormal_metric(
    const float* truth,const float* predicted,int D,int MM)
{
    constexpr int per_lane=(MAX_D_MANOR+31)/32;
    const int lane=threadIdx.x&31;
    double q[MAX_MM_MANOR][per_lane];
    for (int mm=0;mm<MM;++mm) {
        for (int r=0;r<per_lane;++r) {
            const int d=lane+32*r;
            q[mm][r]=d<D ? (double)predicted[mm*D+d] : 0.0;
        }
        // Two-pass modified Gram-Schmidt preserves the supplied subspace;
        // it does not replace the eigensolver or refit the model.
        for (int pass=0;pass<2;++pass) {
            for (int prev=0;prev<mm;++prev) {
                double local=0.0;
                for (int r=0;r<per_lane;++r) local+=q[mm][r]*q[prev][r];
                const double dot=manor_devtan_warp_sum(local);
                for (int r=0;r<per_lane;++r) q[mm][r]-=dot*q[prev][r];
            }
        }
        double local_norm=0.0;
        for (int r=0;r<per_lane;++r) local_norm+=q[mm][r]*q[mm][r];
        const double norm2=manor_devtan_warp_sum(local_norm);
        // A rank-deficient/nonfinite predicted basis is an error, not a zero
        // DevTan. The branch is warp-uniform because norm2 is broadcast.
        if (!(norm2>1e-30) || !isfinite(norm2)) return nanf("");
        const double scale=1.0/sqrt(norm2);
        for (int r=0;r<per_lane;++r) q[mm][r]*=scale;
    }
    double frob2=0.0;
    for (int actual=0;actual<MM;++actual) {
        for (int mm=0;mm<MM;++mm) {
            double local=0.0;
            for (int r=0;r<per_lane;++r) {
                const int d=lane+32*r;
                if (d<D) local+=(double)truth[actual*D+d]*q[mm][r];
            }
            const double dot=manor_devtan_warp_sum(local);
            frob2+=dot*dot;
        }
    }
    if (!isfinite(frob2)) return nanf("");
    return (float)sqrt(fmax(0.0,(double)MM-frob2));
}
