import { Mesh, buildAdjacency, vertexCount, faceCount, getVertex, meshBounds } from './mesh';
import { Vec3, v3, mat4FromBasis, applyMat4, Mat4 } from './math';

/** 解剖标记点（存世界坐标 + 吸附到的最近顶点） */
export interface Landmark {
  id: string;
  label: string;
  position: Vec3;
  vertexId: number;
}

export interface ProcessingReport {
  removedComponents: number;
  removedVertices: number;
  removedFaces: number;
  holesFilled: number;
  trianglesAdded: number;
}

/** 找边界边（仅出现一次的边），返回边界环（顶点 id 序列） */
export function findBoundaryLoops(m: Mesh): number[][] {
  const edges = new Map<string, [number, number]>();
  for (let f = 0; f < faceCount(m); f++) {
    const tri = [m.indices[f * 3], m.indices[f * 3 + 1], m.indices[f * 3 + 2]];
    for (let k = 0; k < 3; k++) {
      const a = tri[k];
      const b = tri[(k + 1) % 3];
      const lo = Math.min(a, b);
      const hi = Math.max(a, b);
      const e = `${lo}|${hi}`;
      const rec = edges.get(e);
      if (!rec) edges.set(e, [lo, hi]);
      else rec[0] = -1; // 第二次出现 → 内部边
    }
  }
  const adj = new Map<number, number[]>();
  for (const [, e] of edges) {
    if (e[0] < 0) continue;
    if (!adj.has(e[0])) adj.set(e[0], []);
    if (!adj.has(e[1])) adj.set(e[1], []);
    adj.get(e[0])!.push(e[1]);
    adj.get(e[1])!.push(e[0]);
  }
  const loops: number[][] = [];
  const visited = new Set<number>();
  for (const start of adj.keys()) {
    if (visited.has(start)) continue;
    const loop: number[] = [];
    let cur = start;
    while (cur !== -1 && !visited.has(cur)) {
      visited.add(cur);
      loop.push(cur);
      let nxt = -1;
      for (const b of adj.get(cur) ?? []) {
        if (!visited.has(b)) { nxt = b; break; }
      }
      cur = nxt;
    }
    if (loop.length >= 3) loops.push(loop);
  }
  return loops;
}

/**
 * 补洞：每个边界环加一个质心顶点并扇形三角化。
 * 对扫描破洞（通常几十到几百条边）足够稳健，法线在后续统一重算。
 */
export function fillHoles(m: Mesh): { holes: number; trianglesAdded: number } {
  const loops = findBoundaryLoops(m);
  if (loops.length === 0) return { holes: 0, trianglesAdded: 0 };
  const pos: number[] = Array.from(m.positions);
  const idx: number[] = Array.from(m.indices);
  let added = 0;
  for (const loop of loops) {
    let cx = 0, cy = 0, cz = 0;
    for (const v of loop) {
      cx += pos[v * 3]; cy += pos[v * 3 + 1]; cz += pos[v * 3 + 2];
    }
    cx /= loop.length; cy /= loop.length; cz /= loop.length;
    const center = pos.length / 3;
    pos.push(cx, cy, cz);
    for (let k = 0; k < loop.length; k++) {
      idx.push(loop[k], loop[(k + 1) % loop.length], center);
      added++;
    }
  }
  m.positions = new Float32Array(pos);
  m.indices = new Uint32Array(idx);
  return { holes: loops.length, trianglesAdded: added };
}

/** 顶点连通域（并查集） */
function connectedComponents(m: Mesh): { label: Uint32Array; sizes: number[]; count: number } {
  const V = vertexCount(m);
  const parent = new Int32Array(V);
  for (let i = 0; i < V; i++) parent[i] = i;
  const find = (x: number): number => {
    while (parent[x] !== x) { parent[x] = parent[parent[x]]; x = parent[x]; }
    return x;
  };
  const union = (a: number, b: number) => {
    const ra = find(a), rb = find(b);
    if (ra !== rb) parent[rb] = ra;
  };
  for (let i = 0; i < m.indices.length; i += 3) {
    union(m.indices[i], m.indices[i + 1]);
    union(m.indices[i + 1], m.indices[i + 2]);
  }
  const rootMap = new Map<number, number>();
  const label = new Uint32Array(V);
  const sizes: number[] = [];
  for (let v = 0; v < V; v++) {
    const r = find(v);
    let id = rootMap.get(r);
    if (id === undefined) { id = sizes.length; rootMap.set(r, id); sizes.push(0); }
    label[v] = id;
    sizes[id]++;
  }
  return { label, sizes, count: sizes.length };
}

/**
 * 扫描清理：
 * 1) 删除小于阈值的连通域（扶架杂点、漂浮碎片、脱落的手臂段）
 * 2) 相对身体轴的截面半径修剪（连着躯干扫进来的手臂/支架），再补洞
 */
export function cleanScan(input: Mesh, opts: { minComponentRatio?: number; radiusFactor?: number; axis?: Vec3 } = {}): { mesh: Mesh; report: ProcessingReport } {
  const minRatio = opts.minComponentRatio ?? 0.02;
  const radiusFactor = opts.radiusFactor ?? 1.55;
  let m: Mesh = { positions: new Float32Array(input.positions), indices: new Uint32Array(input.indices) };

  // ---- 1. 小连通域 ----
  const cc = connectedComponents(m);
  const V0 = vertexCount(m);
  const keep = new Set<number>();
  cc.sizes.forEach((s, id) => { if (s >= V0 * minRatio || id === largestId(cc.sizes)) keep.add(id); });
  const alive = new Uint8Array(V0);
  let removedV = 0;
  for (let v = 0; v < V0; v++) {
    alive[v] = keep.has(cc.label[v]) ? 1 : 0;
    if (!alive[v]) removedV++;
  }
  const keptFaces: number[] = [];
  let removedF = 0;
  for (let f = 0; f < faceCount(m); f++) {
    const a = m.indices[f * 3], b = m.indices[f * 3 + 1], c = m.indices[f * 3 + 2];
    if (alive[a] && alive[b] && alive[c]) keptFaces.push(a, b, c);
    else removedF++;
  }
  m = compactMesh(m, alive, keptFaces);

  // ---- 2. 半径修剪（手臂等突出物）----
  const axis = v3.norm(opts.axis ?? estimateUpAxis(m));
  const bounds = meshBounds(m);
  const origin = bounds.center;
  // 按高度分桶统计中位半径
  const bins = 48;
  const buckets: number[][] = Array.from({ length: bins }, () => []);
  const V1 = vertexCount(m);
  const radial = new Float32Array(V1);
  for (let v = 0; v < V1; v++) {
    const p = getVertex(m, v);
    const h = v3.dot(v3.sub(p, origin), axis);
    const t = (h - v3.dot(v3.sub(bounds.min, origin), axis));
    const range = v3.dot(v3.sub(bounds.max, bounds.min), axis) || 1;
    const bin = Math.min(bins - 1, Math.max(0, Math.floor((t / range) * bins)));
    const perp = v3.sub(v3.sub(p, origin), v3.scale(axis, h));
    const r = v3.len(perp);
    radial[v] = r;
    buckets[bin].push(r);
  }
  const median = buckets.map((b) => {
    if (!b.length) return Infinity;
    const s = [...b].sort((x, y) => x - y);
    return s[Math.floor(s.length / 2)];
  });
  const alive2 = new Uint8Array(V1).fill(1);
  for (let v = 0; v < V1; v++) {
    const p = getVertex(m, v);
    const h = v3.dot(v3.sub(p, origin), axis);
    const t = h - v3.dot(v3.sub(bounds.min, origin), axis);
    const range = v3.dot(v3.sub(bounds.max, bounds.min), axis) || 1;
    const bin = Math.min(bins - 1, Math.max(0, Math.floor((t / range) * bins)));
    if (radial[v] > median[bin] * radiusFactor && isFinite(median[bin])) {
      alive2[v] = 0;
      removedV++;
    }
  }
  const keptFaces2: number[] = [];
  for (let f = 0; f < faceCount(m); f++) {
    const a = m.indices[f * 3], b = m.indices[f * 3 + 1], c = m.indices[f * 3 + 2];
    if (alive2[a] && alive2[b] && alive2[c]) keptFaces2.push(a, b, c);
    else removedF++;
  }
  m = compactMesh(m, alive2, keptFaces2);

  // ---- 3. 补洞 ----
  const fill = fillHoles(m);
  return {
    mesh: m,
    report: {
      removedComponents: cc.count - keep.size,
      removedVertices: removedV,
      removedFaces: removedF,
      holesFilled: fill.holes,
      trianglesAdded: fill.trianglesAdded,
    },
  };
}

function largestId(sizes: number[]): number {
  let id = 0;
  for (let i = 1; i < sizes.length; i++) if (sizes[i] > sizes[id]) id = i;
  return id;
}

/** 按存活掩码压缩网格，重映射索引 */
function compactMesh(m: Mesh, alive: Uint8Array, keptFaces: number[]): Mesh {
  const V = alive.length;
  const remap = new Int32Array(V).fill(-1);
  const pos: number[] = [];
  for (let v = 0; v < V; v++) {
    if (alive[v]) {
      remap[v] = pos.length / 3;
      pos.push(m.positions[v * 3], m.positions[v * 3 + 1], m.positions[v * 3 + 2]);
    }
  }
  const idx = keptFaces.map((v) => remap[v]);
  return {
    positions: new Float32Array(pos),
    indices: new Uint32Array(idx),
    name: m.name,
  };
}

/** PCA 估计躯干竖直轴（标记点缺失时兜底） */
export function estimateUpAxis(m: Mesh): Vec3 {
  const n = vertexCount(m);
  const mean: Vec3 = [0, 0, 0];
  for (let v = 0; v < n; v++) {
    const p = getVertex(m, v);
    mean[0] += p[0]; mean[1] += p[1]; mean[2] += p[2];
  }
  mean[0] /= n; mean[1] /= n; mean[2] /= n;
  const cov = new Float32Array(9);
  for (let v = 0; v < n; v++) {
    const d = v3.sub(getVertex(m, v), mean);
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) cov[i * 3 + j] += d[i] * d[j];
  }
  // 最大特征值对应主方向（躯干最长维度）
  let best = 0;
  for (let i = 1; i < 3; i++) if (cov[i * 3 + i] > cov[best * 3 + best]) best = i;
  const axis: Vec3 = [cov[best], cov[best + 3], cov[best + 6]];
  return v3.norm(axis);
}

/**
 * 由标记点计算摆正变换。
 * 约定：+X 向左（患者左侧），+Y 竖直向上，+Z 向前（腹侧）。
 * 需要左右髂前上棘（asisL/asisR）与左右腋窝点（axillaL/axillaR）。
 */
export function alignmentTransform(landmarks: Landmark[]): Mat4 {
  const get = (id: string) => landmarks.find((l) => l.id === id);
  const asisL = get('asisL')?.position;
  const asisR = get('asisR')?.position;
  const axL = get('axillaL')?.position;
  const axR = get('axillaR')?.position;
  if (!asisL || !asisR || !axL || !axR) throw new Error('标记点不完整，无法摆正（需要双侧髂前上棘与腋窝点）');
  const iliacMid = v3.scale(v3.add(asisL, asisR), 0.5);
  const axillaMid = v3.scale(v3.add(axL, axR), 0.5);
  const xAxis = v3.norm(v3.sub(asisL, asisR)); // 左为 +X
  let yAxis = v3.sub(axillaMid, iliacMid);
  yAxis = v3.norm(v3.sub(yAxis, v3.scale(xAxis, v3.dot(yAxis, xAxis)))); // 正交化
  let zAxis = v3.norm(v3.cross(xAxis, yAxis));
  return mat4FromBasis(xAxis, yAxis, zAxis, iliacMid);
}

/** 应用变换到网格（就地）与标记点 */
export function applyTransform(m: Mesh, t: Mat4): void {
  for (let i = 0; i < m.positions.length; i += 3) {
    const p = applyMat4(t, [m.positions[i], m.positions[i + 1], m.positions[i + 2]]);
    m.positions[i] = p[0]; m.positions[i + 1] = p[1]; m.positions[i + 2] = p[2];
  }
}

/** 最近顶点查询（标记点吸附/拖动用） */
export function nearestVertex(m: Mesh, p: Vec3): { id: number; distance: number } {
  let best = -1;
  let bestD = Infinity;
  for (let v = 0; v < vertexCount(m); v++) {
    const d = v3.dist(getVertex(m, v), p);
    if (d < bestD) { bestD = d; best = v; }
  }
  return { id: best, distance: bestD };
}
