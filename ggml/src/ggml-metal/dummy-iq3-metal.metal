#include <metal_stdlib>
using namespace metal;

kernel void dummy_kernel(device float *output [[ buffer(0) ]],
                         uint tid [[ thread_position_in_grid ]]) {
    float x = 1.0f;
    for (uint i = 0; i < 1000000; ++i) {
        x += i * 0.000001f;
    }
    output[tid] = x;
}
