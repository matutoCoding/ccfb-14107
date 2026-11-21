// 穿戴评估：原始扫描顶点 -> 壳体内壁最近距离（GPU 均匀网格加速）
// 输出带符号间隙（外正内负：穿透为负），以及接触压力估计
struct FitParams {
  n: u32,             // 扫描顶点数
  faceCount: u32,     // 壳体面数
  cellCount: u32,     // 网格桶总数
  toleranceKpa: f32,  // 皮肤耐受阈值 kPa
};

struct Cell {
  start: u32,
  count: u32,
};

@group(0) @binding(0) var<uniform> P: FitParams;
@group(0) @binding(1) var<storage, read> scanPos: array<f32>;
@group(0) @binding(2) var<storage, read> shellPos: array<f32>;
@group(0) @binding(3) var<storage, read> shellIdx: array<u32>;
@group(0) @binding(4) var<storage, read> shellNormals: array<f32>;
@group(0) @binding(5) var<storage, read> cells: array<Cell>;
@group(0) @binding(6) var<storage, read> cellFaces: array<u32>;
@group(0) @binding(7) var<uniform> gridOrigin: vec4<f32>;   // xyz + cellSize
@group(0) @binding(8) var<uniform> gridDim: vec4<u32>;      // nx,ny,nz
@group(0) @binding(9) var<storage, read_write> gap: array<f32>;       // 带符号间隙 mm
@group(0) @binding(10) var<storage, read_write> pressure: array<f32>; // kPa
@group(0) @binding(11) var<storage, read_write> nearestFace: array<u32>;

fn cellIndex(ix: u32, iy: u32, iz: u32) -> u32 {
  return (ix * gridDim.y + iy) * gridDim.z + iz;
}

// 点到三角形最近距离（Ericson），返回距离平方
fn pointTriSq(q: vec3<f32>, a: vec3<f32>, b: vec3<f32>, c: vec3<f32>) -> f32 {
  let ab = b - a;
  let ac = c - a;
  let ap = q - a;
  let d1 = dot(ab, ap);
  let d2 = dot(ac, ap);
  if (d1 <= 0.0 && d2 <= 0.0) { return dot(q - a, q - a); }
  let bp = q - b;
  let d3 = dot(ab, bp);
  let d4 = dot(ac, bp);
  if (d3 >= 0.0 && d4 <= d3) { return dot(q - b, q - b); }
  let vc = d1 * d4 - d3 * d2;
  if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {
    let v = d1 / (d1 - d3);
    let pt = a + v * ab;
    return dot(q - pt, q - pt);
  }
  let cp = q - c;
  let d5 = dot(ab, cp);
  let d6 = dot(ac, cp);
  if (d6 >= 0.0 && d5 <= d6) { return dot(q - c, q - c); }
  let vb = d5 * d2 - d1 * d6;
  if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
    let w = d2 / (d2 - d6);
    let pt = a + w * ac;
    return dot(q - pt, q - pt);
  }
  let va = d3 * d6 - d5 * d4;
  if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
    let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
    let pt = b + w * (c - b);
    return dot(q - pt, q - pt);
  }
  let denom = 1.0 / max(va + vb + vc, 1e-12);
  let v = vb * denom;
  let w = vc * denom;
  let pt = a + ab * v + ac * w;
  return dot(q - pt, q - pt);
}

@compute @workgroup_size(64)
fn fitDistance(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= P.n) { return; }
  let q = vec3<f32>(scanPos[i * 3u], scanPos[i * 3u + 1u], scanPos[i * 3u + 2u]);
  let cellSize = gridOrigin.w;
  var base = vec3<i32>(i32(floor((q - gridOrigin.xyz) / cellSize)));
  var bestSq = 3.402823e38;
  var bestFace = 0u;
  // 螺旋扩张搜索邻接桶，直到找到命中且桶半径超过当前最近距离
  var radius = 0;
  loop {
    if (radius > 12) { break; }
    var foundAny = false;
    for (var dz = -radius; dz <= radius; dz = dz + 1) {
      for (var dy = -radius; dy <= radius; dy = dy + 1) {
        for (var dx = -radius; dx <= radius; dx = dx + 1) {
          if (max(abs(dx), max(abs(dy), abs(dz))) != radius) { continue; }
          let cx = base.x + dx;
          let cy = base.y + dy;
          let cz = base.z + dz;
          if (cx < 0 || cy < 0 || cz < 0) { continue; }
          let ux = u32(cx);
          let uy = u32(cy);
          let uz = u32(cz);
          if (ux >= gridDim.x || uy >= gridDim.y || uz >= gridDim.z) { continue; }
          let cell = cells[cellIndex(ux, uy, uz)];
          if (cell.count == 0u) { continue; }
          foundAny = true;
          for (var k = 0u; k < cell.count; k = k + 1u) {
            let f = cellFaces[cell.start + k];
            let o = f * 3u;
            let ia = shellIdx[o];
            let ib = shellIdx[o + 1u];
            let ic = shellIdx[o + 2u];
            let pa = vec3<f32>(shellPos[ia * 3u], shellPos[ia * 3u + 1u], shellPos[ia * 3u + 2u]);
            let pb = vec3<f32>(shellPos[ib * 3u], shellPos[ib * 3u + 1u], shellPos[ib * 3u + 2u]);
            let pc = vec3<f32>(shellPos[ic * 3u], shellPos[ic * 3u + 1u], shellPos[ic * 3u + 2u]);
            let dsq = pointTriSq(q, pa, pb, pc);
            if (dsq < bestSq) {
              bestSq = dsq;
              bestFace = f;
            }
          }
        }
      }
    }
    // 命中且当前搜索半径（mm）已超过最近距离，可提前退出
    if (foundAny && f32(radius) * cellSize > sqrt(bestSq)) { break; }
    radius = radius + 1;
  }
  let dist = sqrt(bestSq);
  // 符号：扫描点相对壳面法向，内侧（与法向反向）为穿透
  let o = bestFace * 3u;
  let ia = shellIdx[o];
  let nrm = vec3<f32>(shellNormals[ia * 3u], shellNormals[ia * 3u + 1u], shellNormals[ia * 3u + 2u]);
  let pa = vec3<f32>(shellPos[ia * 3u], shellPos[ia * 3u + 1u], shellPos[ia * 3u + 2u]);
  let side = dot(q - pa, nrm);
  let signedGap = select(-dist, dist, side >= 0.0);
  gap[i] = signedGap;
  nearestFace[i] = bestFace;
  // 接触压力简化模型：P = E*t/delta 的弹性基础近似（PETG E≈2.1GPa, 用经验线性标定）
  // 临床经验式：压力(kPa) ≈ k * 穿透深度(mm)，k 由支具材料标定，默认 12 kPa/mm
  if (signedGap < 0.0) {
    pressure[i] = -signedGap * 12.0;
  } else {
    pressure[i] = 0.0;
  }
}
