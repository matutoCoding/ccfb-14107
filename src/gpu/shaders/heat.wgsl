// 热测地法（Crane, Weischedel, Wardetzky 2013）GPU 实现
// 隐式扩散 (M+tL)u=delta -> 单位化负梯度 -> 散度 -> 解 L phi = div -> phi 即测地距离
struct HeatParams {
  n: u32,
  faceCount: u32,
  t: f32,
  _pad: f32,
};

@group(0) @binding(0) var<uniform> P: HeatParams;
@group(0) @binding(1) var<storage, read> positions: array<f32>;
@group(0) @binding(2) var<storage, read> indices: array<u32>;
@group(0) @binding(3) var<storage, read> vArea: array<f32>;
@group(0) @binding(4) var<storage, read_write> heat: array<f32>;
@group(0) @binding(5) var<storage, read_write> gradX: array<f32>;
@group(0) @binding(6) var<storage, read_write> gradY: array<f32>;
@group(0) @binding(7) var<storage, read_write> gradZ: array<f32>;
@group(0) @binding(8) var<storage, read_write> divergence: array<f32>;
@group(0) @binding(9) var<storage, read_write> distance: array<f32>;
@group(0) @binding(10) var<storage, read> faceArea: array<f32>;
@group(0) @binding(11) var<storage, read_write> divergenceInt: array<atomic<i32>>;

fn quantize(x: f32) -> i32 {
  return i32(clamp(x * 100000.0, -2.0e9, 2.0e9));
}

// 扩散源：源顶点（heat 标记为 -1）置 1/面积，其余清零
@compute @workgroup_size(64)
fn initSource(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= P.n) { return; }
  if (heat[i] < -0.5) {
    heat[i] = 1.0 / max(vArea[i], 1e-12);
  } else {
    heat[i] = 0.0;
  }
}

// 单位化负面梯度（指向热源）
@compute @workgroup_size(64)
fn faceGradient(@builtin(global_invocation_id) gid: vec3<u32>) {
  let f = gid.x;
  if (f >= P.faceCount) { return; }
  let o = f * 3u;
  let ia = indices[o];
  let ib = indices[o + 1u];
  let ic = indices[o + 2u];
  let pa = vec3<f32>(positions[ia * 3u], positions[ia * 3u + 1u], positions[ia * 3u + 2u]);
  let pb = vec3<f32>(positions[ib * 3u], positions[ib * 3u + 1u], positions[ib * 3u + 2u]);
  let pc = vec3<f32>(positions[ic * 3u], positions[ic * 3u + 1u], positions[ic * 3u + 2u]);
  let e_ab = pb - pa;
  let e_bc = pc - pb;
  let e_ca = pa - pc;
  let nrm = normalize(cross(e_ab, -e_ca));
  let area2 = max(2.0 * faceArea[f], 1e-12);
  let g = (heat[ia] * cross(nrm, e_ca) + heat[ib] * cross(nrm, e_ab)
         + heat[ic] * cross(nrm, e_bc)) / area2;
  let gl = max(length(g), 1e-12);
  let gn = -g / gl;
  gradX[f] = gn.x;
  gradY[f] = gn.y;
  gradZ[f] = gn.z;
}

fn cotAt(opp: vec3<f32>, q1: vec3<f32>, q2: vec3<f32>) -> f32 {
  let u = q1 - opp;
  let v = q2 - opp;
  let d = max(length(cross(u, v)), 1e-12);
  return max(1e-6, dot(u, v) / d);
}

// 面散度：定点原子累加到顶点
@compute @workgroup_size(64)
fn faceDivergence(@builtin(global_invocation_id) gid: vec3<u32>) {
  let f = gid.x;
  if (f >= P.faceCount) { return; }
  let o = f * 3u;
  let ia = indices[o];
  let ib = indices[o + 1u];
  let ic = indices[o + 2u];
  let pa = vec3<f32>(positions[ia * 3u], positions[ia * 3u + 1u], positions[ia * 3u + 2u]);
  let pb = vec3<f32>(positions[ib * 3u], positions[ib * 3u + 1u], positions[ib * 3u + 2u]);
  let pc = vec3<f32>(positions[ic * 3u], positions[ic * 3u + 1u], positions[ic * 3u + 2u]);
  let X = vec3<f32>(gradX[f], gradY[f], gradZ[f]);
  let A = faceArea[f];
  let cotA = cotAt(pa, pb, pc);
  let cotB = cotAt(pb, pc, pa);
  let cotC = cotAt(pc, pa, pb);
  let dA = A * (cotB * dot(pa - pc, X) + cotC * dot(pa - pb, X));
  let dB = A * (cotC * dot(pb - pa, X) + cotA * dot(pb - pc, X));
  let dC = A * (cotA * dot(pc - pb, X) + cotB * dot(pc - pa, X));
  atomicAdd(&divergenceInt[ia], quantize(dA));
  atomicAdd(&divergenceInt[ib], quantize(dB));
  atomicAdd(&divergenceInt[ic], quantize(dC));
}

@compute @workgroup_size(64)
fn dequantizeDivergence(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= P.n) { return; }
  divergence[i] = f32(atomicLoad(&divergenceInt[i])) / 100000.0;
  atomicStore(&divergenceInt[i], 0);
}

// 距离平移归一化（minDist 由 CPU 读回后写入 uniform.t 复用参数）
@compute @workgroup_size(64)
fn shiftDistance(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= P.n) { return; }
  // phi 已由外部拷贝到 distance，平移使其最小值为 0
  distance[i] = max(0.0, distance[i] - P.t);
}
