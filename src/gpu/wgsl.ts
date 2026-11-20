/**
 * 自写 WGSL 计算着色器。所有核函数与 CPU 参考实现（src/geometry）同构，
 * 在无 WebGPU 的环境由 GpuCompute 自动回退到 CPU。
 */

/** 第一遍：每个三角形写一条面积加权（未归一化）法线到面法线缓冲（无原子） */
export const FACE_NORMALS_WGSL = /* wgsl */ `
struct TriMesh {
  faceCount: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
};
@group(0) @binding(0) var<uniform> meta: TriMesh;
@group(0) @binding(1) var<storage, read> positions: array<f32>;
@group(0) @binding(2) var<storage, read> indices: array<u32>;
@group(0) @binding(3) var<storage, read_write> faceNormals: array<f32>;

fn loadPos(v: u32) -> vec3<f32> {
  return vec3<f32>(positions[v*3], positions[v*3+1], positions[v*3+2]);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let f = gid.x;
  if (f >= meta.faceCount) { return; }
  let base = f * 3u;
  let a = indices[base];
  let b = indices[base + 1u];
  let c = indices[base + 2u];
  let n = cross(loadPos(b) - loadPos(a), loadPos(c) - loadPos(a));
  faceNormals[f*3] = n.x;
  faceNormals[f*3+1] = n.y;
  faceNormals[f*3+2] = n.z;
}
`;

/** 第二遍：用顶点-面 CSR 关联累加面积加权面法线并归一化 */
export const VERTEX_NORMALS_WGSL = /* wgsl */ `
struct VMeta { vertexCount: u32, _a: u32, _b: u32, _c: u32 };
@group(0) @binding(0) var<uniform> meta: VMeta;
@group(0) @binding(1) var<storage, read> faceNormals: array<f32>;
@group(0) @binding(2) var<storage, read> faceOffset: array<u32>;
@group(0) @binding(3) var<storage, read> vertexFaces: array<u32>;
@group(0) @binding(4) var<storage, read_write> normals: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let v = gid.x;
  if (v >= meta.vertexCount) { return; }
  var acc = vec3<f32>(0.0, 0.0, 0.0);
  let start = faceOffset[v];
  let end = faceOffset[v + 1u];
  for (var j = start; j < end; j = j + 1u) {
    let f = vertexFaces[j];
    acc = acc + vec3<f32>(faceNormals[f*3], faceNormals[f*3+1], faceNormals[f*3+2]);
  }
  let l = length(acc);
  let inv = select(0.0, 1.0 / l, l > 1e-9);
  normals[v*3] = acc.x * inv;
  normals[v*3+1] = acc.y * inv;
  normals[v*3+2] = acc.z * inv;
}
`;

/**
 * 笔刷位移核：输入测地距离场，按高斯衰减沿法线推/拉顶点。
 * sign=-1 压垫（沿 -N），sign=+1 释放（沿 +N）。
 */
export const BRUSH_DISPLACE_WGSL = /* wgsl */ `
struct Brush {
  centerVertex: u32,
  sign: f32,
  radiusMm: f32,
  depthMm: f32,
};
@group(0) @binding(0) var<uniform> brush: Brush;
@group(0) @binding(1) var<storage, read> distances: array<f32>;
@group(0) @binding(2) var<storage, read> normals: array<f32>;
@group(0) @binding(3) var<storage, read_write> positions: array<f32>;
@group(0) @binding(4) var<storage, read_write> displacement: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let v = gid.x;
  let count = arrayLength(&distances);
  if (v >= count) { return; }
  let d = distances[v];
  if (d > brush.radiusMm) { displacement[v] = 0.0; return; }
  let sigma = brush.radiusMm / 3.0;
  let w = exp(-(d*d) / (2.0 * sigma * sigma));
  let disp = brush.sign * brush.depthMm * w;
  displacement[v] = disp;
  let n = vec3<f32>(normals[v*3], normals[v*3+1], normals[v*3+2]);
  positions[v*3] = positions[v*3] + n.x * disp;
  positions[v*3+1] = positions[v*3+1] + n.y * disp;
  positions[v*3+2] = positions[v*3+2] + n.z * disp;
}
`;

/** 等距偏移核：沿顶点法线外偏 thickness(v)，生成外壁位置 */
export const OFFSET_SURFACE_WGSL = /* wgsl */ `
struct OMeta { vertexCount: u32, _a: u32, _b: u32, _c: u32 };
@group(0) @binding(0) var<uniform> meta: OMeta;
@group(0) @binding(1) var<storage, read> innerPos: array<f32>;
@group(0) @binding(2) var<storage, read> normals: array<f32>;
@group(0) @binding(3) var<storage, read> thickness: array<f32>;
@group(0) @binding(4) var<storage, read_write> outerPos: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let v = gid.x;
  if (v >= meta.vertexCount) { return; }
  let n = vec3<f32>(normals[v*3], normals[v*3+1], normals[v*3+2]);
  let t = thickness[v];
  outerPos[v*3] = innerPos[v*3] + n.x * t;
  outerPos[v*3+1] = innerPos[v*3+1] + n.y * t;
  outerPos[v*3+2] = innerPos[v*3+2] + n.z * t;
}
`;

/**
 * Jacobi 热扩散迭代（GPU 热法第一步）。
 * 求解 (M + tL) u = δ，L 为伞状（对角 degree、非对角 -1）。
 * 每轮 u_new[i] = (rhs[i] + t·Σ_neighbor u[j]) / (area[i] + t·degree[i])。
 * CSR 邻接通过 offset/neighbors 访问。
 */
export const HEAT_JACOBI_WGSL = /* wgsl */ `
struct HMeta {
  vertexCount: u32,
  timeStep: f32,
  _a: u32,
  _b: u32,
};
@group(0) @binding(0) var<uniform> meta: HMeta;
@group(0) @binding(1) var<storage, read> cur: array<f32>;
@group(0) @binding(2) var<storage, read_write> nxt: array<f32>;
@group(0) @binding(3) var<storage, read> areas: array<f32>;
@group(0) @binding(4) var<storage, read> degree: array<u32>;
@group(0) @binding(5) var<storage, read> offset: array<u32>;
@group(0) @binding(6) var<storage, read> neighbors: array<u32>;
@group(0) @binding(7) var<storage, read> rhs: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= meta.vertexCount) { return; }
  var sum = 0.0;
  let start = offset[i];
  let end = offset[i + 1u];
  for (var j = start; j < end; j = j + 1u) {
    sum = sum + cur[neighbors[j]];
  }
  let diag = areas[i] + meta.timeStep * f32(degree[i]);
  nxt[i] = (rhs[i] + meta.timeStep * sum) / diag;
}
`;
