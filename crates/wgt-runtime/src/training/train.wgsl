// One 64-lane workgroup covers 64 independent destinations, flattened row-major.
// Linear: lane = (sample, output); dX: (sample, input); dW: (output, input).
// Each dW/db lane reduces the batch in increasing sample order. Separate dispatches
// accumulate layer uses and fold view/add gradients; no floating-point atomics.
// Tape, gradients and candidate weights occupy disjoint ranges in one f32 arena.
struct Config {
    operation: u32, count: u32, input_width: u32, output_width: u32,
    batch: u32, a: u32, b: u32, c: u32,
    destination: u32, extra: u32, rate: f32, padding: u32,
}
@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<storage, read_write> status: atomic<u32>;
@group(0) @binding(2) var<uniform> cfg: Config;

// WGSL does not promise portable NaN/Inf propagation. Detect overflow before
// arithmetic instead of relying solely on is-finite checks after GPU readback.
fn add(a: f32, b: f32) -> f32 {
    let limit = 3.4028234663852886e38;
    if (a > 0.0 && b > 0.0 && a > limit - b) ||
       (a < 0.0 && b < 0.0 && a < -limit - b) {
        atomicOr(&status, 1u); return 0.0;
    }
    return a + b;
}
fn mul(a: f32, b: f32) -> f32 {
    let limit = 3.4028234663852886e38;
    if abs(b) > 1.0 && abs(a) > limit / abs(b) {
        atomicOr(&status, 1u); return 0.0;
    }
    if abs(a) > 1.0 && abs(b) > limit / abs(a) {
        atomicOr(&status, 1u); return 0.0;
    }
    return a * b;
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let j = id.x;
    if j >= cfg.count { return; }
    var value = 0.0;
    switch cfg.operation {
        case 0u: { // Y[B,O] = X[B,I] W[O,I]^T + bias[O]
            let row = j % cfg.output_width;
            let sample = j / cfg.output_width;
            if cfg.extra != 0u { value = arena[cfg.c + row]; }
            for (var k = 0u; k < cfg.input_width; k++) {
                value = add(value, mul(arena[cfg.a + sample * cfg.input_width + k],
                                       arena[cfg.b + row * cfg.input_width + k]));
            }
        }
        case 1u, 2u, 3u, 4u: {
            let x = arena[cfg.a + j];
            value = x;
            if cfg.operation == 1u { value = max(x, 0.0); }
            if cfg.operation == 2u { value = tanh(clamp(x, -10.0, 10.0)); }
            if cfg.operation == 3u {
                // Stable sigmoid: exp's argument is always nonpositive.
                let e = exp(-abs(x));
                value = select(e / (1.0 + e), 1.0 / (1.0 + e), x >= 0.0);
            }
        }
        case 5u: { // Per-output squared error, averaged over output width.
            let d = add(arena[cfg.a + j], -arena[cfg.b + j]);
            value = mul(d, d) / f32(cfg.output_width);
        }
        case 6u: { // dMSE/dY, averaged over both batch and output width.
            let d = add(arena[cfg.a + j], -arena[cfg.b + j]);
            value = mul(2.0, d) / f32(cfg.count);
        }
        case 7u: { // Mean loss; serial deterministic reduction, one destination.
            for (var k = 0u; k < cfg.extra; k++) { value = add(value, arena[cfg.a + k]); }
            value /= f32(cfg.batch);
        }
        case 8u: { // dX = dY W (the original weights, never candidate weights).
            let sample = j / cfg.input_width;
            let col = j % cfg.input_width;
            for (var row = 0u; row < cfg.output_width; row++) {
                value = add(value, mul(arena[cfg.a + sample * cfg.output_width + row],
                                      arena[cfg.b + row * cfg.input_width + col]));
            }
        }
        case 9u: { // dW += dY^T X; repeated layer uses accumulate in dispatch order.
            value = arena[cfg.destination + j];
            let row = j / cfg.input_width;
            let col = j % cfg.input_width;
            for (var sample = 0u; sample < cfg.batch; sample++) {
                value = add(value, mul(arena[cfg.a + sample * cfg.output_width + row],
                                      arena[cfg.b + sample * cfg.input_width + col]));
            }
        }
        case 10u: { // db += sum_B(dY)
            value = arena[cfg.destination + j];
            for (var sample = 0u; sample < cfg.batch; sample++) {
                value = add(value, arena[cfg.a + sample * cfg.output_width + j]);
            }
        }
        case 11u, 12u, 13u, 14u: {
            value = arena[cfg.a + j];
            let x = arena[cfg.b + j];
            let y = arena[cfg.c + j];
            if cfg.operation == 11u && x <= 0.0 { value = 0.0; }
            if cfg.operation == 12u { value = mul(value, 1.0 - y * y); }
            if cfg.operation == 13u { value = mul(value, y * (1.0 - y)); }
        }
        case 15u: { value = add(arena[cfg.destination + j], arena[cfg.a + j]); }
        case 16u: { value = 0.0; } // Frozen expression: stop the owner gradient chain.
        case 17u: { value = add(arena[cfg.a + j], -mul(cfg.rate, arena[cfg.b + j])); }
        default: { atomicOr(&status, 2u); }
    }
    arena[cfg.destination + j] = value;
}
