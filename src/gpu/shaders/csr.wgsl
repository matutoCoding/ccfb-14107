// CSR 稀疏矩阵 × 稠密向量
struct Params {
  n: u32,
  _a: f32,
  _b: f32,
  _c: f32,
};

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> csrOff: array<u32>;
@group(0) @binding(2) var<storage, read> csrIdx: array<u32>;
@group(0) @binding(3) var<storage, read> csrVal: array<f32>;
@group(0) @binding(4) var<storage, read> x: array<f32>;
@group(0) @binding(5) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(64)
fn spmv(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.x;
  if (row >= P.n) { return; }
  var sum = 0.0;
  let lo = csrOff[row];
  let hi = csrOff[row + 1u];
  for (var k = lo; k < hi; k = k + 1u) {
    sum = sum + csrVal[k] * x[csrIdx[k]];
  }
  y[row] = sum;
}
