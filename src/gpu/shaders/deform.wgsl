// 修型变形核与等距偏移核
struct DeformParams {
  n: u32,
  brushCount: u32,
  brushIndex: u32,   // applyBrush 用的笔刷下标
  _pad1: f32,
};

struct Brush {
  centerVertex: u32,
  kind: u32,        // 0=压垫(向内) 1=释放区(向外)
  sigma: f32,       // 测地距离标准差 (mm)
  depthMm: f32,     // 推压深度 (mm，带符号)
  _pad: vec2<f32>,
};

@group(0) @binding(0) var<uniform> P: DeformParams;
@group(0) @binding(1) var<storage, read> positions: array<f32>;
@group(0) @binding(2) var<storage, read> normals: array<f32>;
@group(0) @binding(3) var<storage, read> geoDist: array<f32>;   // 当前笔刷中心的测地距离场
@group(0) @binding(4) var<storage, read> brushes: array<Brush>;
@group(0) @binding(5) var<storage, read_write> outPositions: array<f32>;
@group(0) @binding(6) var<storage, read_write> weightSum: array<f32>;

// 单笔刷高斯测地衰减变形：位移沿顶点法向，权重 exp(-d²/2σ²)
@compute @workgroup_size(64)
fn applyBrush(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= P.n) { return; }
  // geoDist 对应当前笔刷，下标由 uniform.brushIndex 给出
  let b = brushes[P.brushIndex];
  let d = geoDist[i];
  let w = exp(-(d * d) / (2.0 * b.sigma * b.sigma));
  let sign = select(1.0, -1.0, b.kind == 0u); // 压垫沿 -n 内推
  let disp = sign * b.depthMm * w;
  outPositions[i * 3u] = positions[i * 3u] + normals[i * 3u] * disp;
  outPositions[i * 3u + 1u] = positions[i * 3u + 1u] + normals[i * 3u + 1u] * disp;
  outPositions[i * 3u + 2u] = positions[i * 3u + 2u] + normals[i * 3u + 2u] * disp;
  weightSum[i] = w;
}

// 多笔刷叠加（在已变形结果上继续累加，权重按方案中所有 dab）
@compute @workgroup_size(64)
fn applyAllBrushes(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= P.n) { return; }
  var px = positions[i * 3u];
  var py = positions[i * 3u + 1u];
  var pz = positions[i * 3u + 2u];
  let nx = normals[i * 3u];
  let ny = normals[i * 3u + 1u];
  let nz = normals[i * 3u + 2u];
  // geoDist 缓冲布局：brushCount 个距离场平铺，每段 n 个 float
  for (var bi = 0u; bi < P.brushCount; bi = bi + 1u) {
    let b = brushes[bi];
    let d = geoDist[bi * P.n + i];
    let w = exp(-(d * d) / (2.0 * b.sigma * b.sigma));
    let sgn = select(1.0, -1.0, b.kind == 0u);
    let disp = sgn * b.depthMm * w;
    px = px + nx * disp;
    py = py + ny * disp;
    pz = pz + nz * disp;
  }
  outPositions[i * 3u] = px;
  outPositions[i * 3u + 1u] = py;
  outPositions[i * 3u + 2u] = pz;
}

// 等距偏移：newPos = pos + normal * thickness(v)
struct OffsetParams {
  n: u32,
  direction: f32,   // +1 外壁，-1 内壁
  _a: f32,
  _b: f32,
};
@group(0) @binding(7) var<uniform> OP: OffsetParams;
@group(0) @binding(8) var<storage, read> thickness: array<f32>;

@compute @workgroup_size(64)
fn offsetSurface(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= OP.n) { return; }
  let t = OP.direction * thickness[i];
  outPositions[i * 3u] = positions[i * 3u] + normals[i * 3u] * t;
  outPositions[i * 3u + 1u] = positions[i * 3u + 1u] + normals[i * 3u + 1u] * t;
  outPositions[i * 3u + 2u] = positions[i * 3u + 2u] + normals[i * 3u + 2u] * t;
}
