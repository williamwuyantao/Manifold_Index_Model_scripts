"""Derive the cached-Gram readout from the unchanged prediction kernel."""
from pathlib import Path
import argparse


def generate(source, output):
    text = source.read_text()
    start = text.index('#if MANOR_M_EVAL_MIN_BLOCKS_PER_SM > 0\n__global__')
    end = text.index('\ntemplate <int MAXD>\n__global__ void owner_aggregation_accumulate_kernel', start)
    kernel = text[start:end]
    replacements = {
        'void owner_fused_eval_tiled_kernel(': 'void multi_cached_gram_eval_kernel(',
        'const float* __restrict__ Xtrainnorm2, const float* __restrict__ TdotX,':
            'const float* __restrict__ Xtrainnorm2, const float* __restrict__ TdotX,\n'
            '    const float* __restrict__ train_gram,',
        '''                    float p = 0.0f;
                    for (int mm = 0; mm < MM; ++mm)
                        p += tj[(size_t)mm * D + r] * tj[(size_t)mm * D + c];
                    pj[t] = p;''':
            '                    pj[t] = train_gram[(size_t)(j0 + jt) * ntri + t];',
    }
    for before, after in replacements.items():
        if kernel.count(before) != 1:
            raise ValueError('Readout source marker changed: '+before[:80])
        kernel = kernel.replace(before, after)
    prepare = r'''
// One original FP32 Gram expression per training point and matrix entry.
// The consumer keeps the original query, tile and accumulation order.
__global__ void multi_prepare_tangent_gram(const float* tangent, int N,
    int D, int MM, float* gram) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int j = blockIdx.x * (blockDim.x >> 5) + warp;
    if (j >= N) return;
    const int ntri = D * (D + 1) / 2;
    const float* tj = tangent + (size_t)j * MM * D;
    for (int t = lane; t < ntri; t += 32) {
        int r = 0, base = 0;
        while (base + (D - r) <= t) { base += (D - r); ++r; }
        const int c = r + (t - base);
        float p = 0.0f;
        for (int mm = 0; mm < MM; ++mm)
            p += tj[(size_t)mm * D + r] * tj[(size_t)mm * D + c];
        gram[(size_t)j * ntri + t] = p;
    }
}
'''
    output.write_text(prepare+kernel)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source', type=Path)
    p.add_argument('output', type=Path)
    a = p.parse_args()
    generate(a.source, a.output)
