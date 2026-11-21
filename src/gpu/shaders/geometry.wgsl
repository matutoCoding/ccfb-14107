// 几何辅助核：面积加权法向（定点原子）、网格 AO、标量色谱
struct GeoParams {
  n: u32,
  faceCount: u32,
  aoRadius: f32,
  aoStrength: f32,
};

@group(0) @binding(0) var<uniform> P: GeoParams;
@group(0) @binding(1) var<storage, read> positions: array<f32>;
@group(0) @binding(2) var<storage, read> indices: array<u32>;
@group(0) @binding(3) var<storage, read_write> normals: array<f32>;
@group(0) @binding(4) var<storage, read> v2vOff: array<u32>;
@group(0) @binding(5) var<storage, read> v2vIdx: array<u32>;
@group(0) @binding(6) var<storage, read_write> ao: array<f32>;
@group(0) @binding(7) var<storage, read> scalar: array<f32>;
@group(0) @binding(8) var<storage, read_write> colors: array<f32>;
@group(0) @binding(9) var<storage, read_write> normalAccInt: array<atomic<i32>>;

struct ColorParams {
  n: u32,
  maxAbs: f32,
  mode: f32,   // 0=差值红蓝 1=压力 2=厚度
  _pad: f32,
};
@group(0) @binding(10) var<uniform> CP: ColorParams;

fn q32(x: f32) -> i32 { return i32(clamp(x * 4096.0, -2.0e9, 2.0e9)); }

// 面法向定点原子累加到顶点
@compute @workgroup_size(64)
fn accumulateFaceNormals(@builtin(global_invocation_id) gid: vec3<u32>) {
  let f = gid.x;
  if (f >= P.faceCount) { return; }
  let o = f * 3u;
  let ia = indices[o];
  let ib = indices[o + 1u];
  let ic = indices[o + 2u];
  let pa = vec3<f32>(positions[ia * 3u], positions[ia * 3u + 1u], positions[ia * 3u + 2u]);
  let pb = vec3<f32>(positions[ib * 3u], positions[ib * 3u + 1u], positions[ib * 3u + 2u]);
  let pc = vec3<f32>(positions[ic * 3u], positions[ic * 3u + 1u], positions[ic * 3u + 2u]);
  let fnv = cross(pb - pa, pc - pa);
  for (var k = 0u; k < 3u; k = k + 1u) {
    let vi = indices[o + k];
    atomicAdd(&normalAccInt[vi * 3u], q32(fnv.x));
    atomicAdd(&normalAccInt[vi * 3u + 1u], q32(fnv.y));
    atomicAdd(&normalAccInt[vi * 3u + 2u], q32(fnv.z));
  }
}

@compute @workgroup_size(64)
fn normalizeVertexNormals(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= P.n) { return; }
  let nx = f32(atomicLoad(&normalAccInt[i * 3u])) / 4096.0;
  let ny = f32(atomicLoad(&normalAccInt[i * 3u + 1u])) / 4096.0;
  let nz = f32(atomicLoad(&normalAccInt[i * 3u + 2u])) / 4096.0;
  let l = max(length(vec3<f32>(nx, ny, nz)), 1e-12);
  normals[i * 3u] = nx / l;
  normals[i * 3u + 1u] = ny / l;
  normals[i * 3u + 2u] = nz / l;
  atomicStore(&normalAccInt[i * 3u], 0);
  atomicStore(&normalAccInt[i * 3u + 1u], 0);
  atomicStore(&normalAccInt[i * 3u + 2u], 0);
}

// 顶点邻域 AO：邻接法向夹角 + 朝向遮挡的加权平均
@compute @workgroup_size(64)
fn computeAO(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= P.n) { return; }
  let pi = vec3<f32>(positions[i * 3u], positions[i * 3u + 1u], positions[i * 3u + 2u]);
  let ni = vec3<f32>(normals[i * 3u], normals[i * 3u + 1u], normals[i * 3u + 2u]);
  var occ = 0.0;
  var wsum = 0.0;
  let lo = v2vOff[i];
  let hi = v2vOff[i + 1u];
  for (var k = lo; k < hi; k = k + 1u) {
    let j = v2vIdx[k];
    let pj = vec3<f32>(positions[j * 3u], positions[j * 3u + 1u], positions[j * 3u + 2u]);
    let v = pj - pi;
    let dist = length(v);
    if (dist < 1e-6 || dist > P.aoRadius) { continue; }
    let w = 1.0 - dist / P.aoRadius;
    let nj = vec3<f32>(normals[j * 3u], normals[j * 3u + 1u], normals[j * 3u + 2u]);
    let facing = 1.0 - clamp(dot(normalize(v), ni), -1.0, 1.0);
    let bend = 1.0 - clamp(dot(ni, nj), -1.0, 1.0);
    occ = occ + w * facing * (0.5 + 0.5 * bend);
    wsum = wsum + w;
  }
  ao[i] = clamp(1.0 - P.aoStrength * occ / max(wsum, 1e-6), 0.0, 1.0);
}

fn rampDiverging(t: f32) -> vec3<f32> {
  let c = select(vec3<f32>(0.1, 0.35, 0.95), vec3<f32>(0.95, 0.25, 0.15), t > 0.0);
  let a = abs(t);
  return mix(vec3<f32>(0.95, 0.95, 0.95), c, a);
}

fn rampPressure(t: f32) -> vec3<f32> {
  let x = clamp(t, 0.0, 1.0);
  if (x < 0.5) {
    return mix(vec3<f32>(0.2, 0.75, 0.3), vec3<f32>(0.98, 0.85, 0.2), x * 2.0);
  }
  return mix(vec3<f32>(0.98, 0.85, 0.2), vec3<f32>(0.9, 0.12, 0.1), (x - 0.5) * 2.0);
}

@compute @workgroup_size(64)
fn scalarToColor(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= CP.n) { return; }
  let s = scalar[i];
  var col = vec3<f32>(0.8, 0.8, 0.8);
  if (CP.mode < 0.5) {
    col = rampDiverging(s / max(CP.maxAbs, 1e-6));
  } else if (CP.mode < 1.5) {
    col = rampPressure(s / max(CP.maxAbs, 1e-6));
  } else {
    col = mix(vec3<f32>(0.2, 0.4, 0.9), vec3<f32>(0.9, 0.3, 0.2), clamp((s - 3.0) / 2.0, 0.0, 1.0));
  }
  colors[i * 3u] = col.x;
  colors[i * 3u + 1u] = col.y;
  colors[i * 3u + 2u] = col.z;
}
