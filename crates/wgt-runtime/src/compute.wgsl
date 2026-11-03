// Each linear workgroup owns an output row. Its lanes cooperatively reduce the
// input axis; activations use one lane per output element. No atomic float adds.
struct Config { input_width: u32, output_width: u32, operation: u32, has_bias: u32 }
@group(0) @binding(0) var<storage, read> source: array<f32>;
@group(0) @binding(1) var<storage, read_write> destination: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<uniform> config: Config;
var<workgroup> partial: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_id) local: vec3<u32>) {
    let lane = local.x;
    if config.operation == 0u {
        let row = group.x;
        var sum = 0.0;
        for (var col = lane; col < config.input_width; col += 64u) {
            sum += source[col] * weights[row * config.input_width + col];
        }
        partial[lane] = sum;
        workgroupBarrier();
        for (var stride = 32u; stride > 0u; stride /= 2u) {
            if lane < stride { partial[lane] += partial[lane + stride]; }
            workgroupBarrier();
        }
        if lane == 0u {
            var value = partial[0];
            if config.has_bias != 0u { value += weights[config.input_width * config.output_width + row]; }
            destination[row] = value;
        }
    } else {
        let index = group.x * 64u + lane;
        if index >= config.output_width { return; }
        let x = source[index];
        var value = x;
        switch config.operation {
            case 1u: { value = max(x, 0.0); }
            // Avoid exp overflow in tanh's implementation on some GPU backends.
            case 2u: { value = tanh(clamp(x, -10.0, 10.0)); }
            case 3u: { value = 1.0 / (1.0 + exp(-x)); }
            default: {}
        }
        destination[index] = value;
    }
}
