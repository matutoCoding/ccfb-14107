// 稠密向量核：axpy / 带目标槽位的点积两阶段归约 / 拷贝
struct VecParams {
  n: u32,
  alpha: f32,
  beta: f32,
  slot: f32,   // dotStage2 写入 scalars 的槽位（整数语义）
};

@group(0) @binding(0) var<uniform> P: VecParams;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<storage, read_write> partial: array<f32>;

// out = alpha * a + beta * b
@compute @workgroup_size(256)
fn axpy2(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= P.n) { return; }
  out[i] = P.alpha * a[i] + P.beta * b[i];
}

var<workgroup> tile: array<f32, 256>;

// 点积阶段 1：每 workgroup 一个部分和
@compute @workgroup_size(256)
fn dotStage1(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let i = gid.x;
  var v = 0.0;
  if (i < P.n) { v = a[i] * b[i]; }
  tile[lid.x] = v;
  workgroupBarrier();
  var s = 128u;
  loop {
    if (s == 0u) { break; }
    if (lid.x < s) { tile[lid.x] = tile[lid.x] + tile[lid.x + s]; }
    workgroupBarrier();
    s = s / 2u;
  }
  if (lid.x == 0u) { partial[gid.y] = tile[0]; }
}

// 点积阶段 2：归约部分和，写入 out[slot]
@compute @workgroup_size(256)
fn dotStage2(@builtin(global_invocation_id) lid: vec3<u32>) {
  let blocks = (P.n + 255u) / 256u;
  var sum = 0.0;
  var base = 0u;
  loop {
    if (base >= blocks) { break; }
    if (base + lid.x < blocks) { sum = sum + partial[base + lid.x]; }
    base = base + 256u;
  }
  tile[lid.x] = sum;
  workgroupBarrier();
  var s = 128u;
  loop {
    if (s == 0u) { break; }
    if (lid.x < s) { tile[lid.x] = tile[lid.x] + tile[lid.x + s]; }
    workgroupBarrier();
    s = s / 2u;
  }
  if (lid.x == 0u) { out[u32(P.slot)] = tile[0]; }
}

@compute @workgroup_size(256)
fn copyVec(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= P.n) { return; }
  out[i] = a[i];
}
