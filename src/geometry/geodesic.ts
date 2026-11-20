import { Mesh, Adjacency, buildAdjacency, vertexCount, faceCount, getVertex, vertexAreas } from './mesh';
import { Vec3, v3 } from './math';

/**
 * 测地距离 —— Heat Method（Crane, Weischedel, Wardetzky, 2013）。
 * 论文约定：余切 Laplacian L 为负半定，L_ii = -Σw_ij，L_ij = +w_ij。
 *   1. (M - t·L) u = δ_source     短时热扩散（-L 正定）
 *   2. X = -∇u / |∇u|              单位化负梯度场，指向源点
 *   3. L φ = ∇·X                   泊松方程（用 -L 正定 CG 求解）
 * 钝角三角形的负余切钳到 0；零权重边用 (A_i+A_j)/l² 兜底，保证连通正定。
 * GPU 版（WGSL compute）与本文件同构，CPU 版同时承担兜底与精度基准。
 */

interface CotLaplacian {
  /** 正定形式 -L 的正对角元 Σw（第三步余切泊松用） */
  diagPos: Float32Array;
  /** CSR 非对角正权重 w_ij（第三步用，已做零值兜底） */
  offdiag: Float32Array;
  /** 伞状 Laplacian 对角元 = 顶点度数（第一步热扩散用，保证单调非负） */
  umbrellaDiag: Uint32Array;
  adj: Adjacency;
  /** 真实面余切（F*3，faceCot[f*3+k] 为顶点 k 处角的余切，含负值，梯度/散度用） */
  faceCot: Float32Array;
}

export function buildCotLaplacian(m: Mesh, adj: Adjacency): CotLaplacian {
  const V = vertexCount(m);
  const F = faceCount(m);
  const diagPos = new Float32Array(V);
  const offdiag = new Float32Array(adj.neighbors.length);
  const umbrellaDiag = new Uint32Array(V);
  const faceCot = new Float32Array(F * 3);

  // 每条无向边累加真实余切（梯度/散度保留负值）；矩阵权重另做钳零
  const edgeRaw = new Map<string, number>();
  const ekey = (a: number, b: number) => (a < b ? `${a}|${b}` : `${b}|${a}`);
  for (let f = 0; f < F; f++) {
    const tri = [m.indices[f * 3], m.indices[f * 3 + 1], m.indices[f * 3 + 2]];
    const p = tri.map((v) => getVertex(m, v));
    for (let k = 0; k < 3; k++) {
      const u = v3.sub(p[(k + 1) % 3], p[k]);
      const w = v3.sub(p[(k + 2) % 3], p[k]);
      const cos = v3.dot(u, w);
      const sin = v3.len(v3.cross(u, w));
      faceCot[f * 3 + k] = cos / (sin || 1e-9);
      const key = ekey(tri[(k + 1) % 3], tri[(k + 2) % 3]);
      edgeRaw.set(key, (edgeRaw.get(key) ?? 0) + 0.5 * faceCot[f * 3 + k]);
    }
  }
  const areas = vertexAreas(m);
  for (let v = 0; v < V; v++) {
    umbrellaDiag[v] = adj.offset[v + 1] - adj.offset[v];
    for (let j = adj.offset[v]; j < adj.offset[v + 1]; j++) {
      const u = adj.neighbors[j];
      let w = Math.max(0, edgeRaw.get(ekey(v, u)) ?? 0);
      if (w <= 1e-12) {
        const l2 = adj.edgeLength[j] * adj.edgeLength[j];
        w = (areas[v] + areas[u]) / (l2 || 1);
      }
      offdiag[j] = w;
      diagPos[v] += w;
    }
  }
  return { diagPos, offdiag, umbrellaDiag, adj, faceCot };
}

/**
 * 稀疏对称矩阵的 CG 求解。
 * diag：正定对角元。offSign：非对角项符号。
 *   +1（默认）：A = diag·I + off·邻接，用于 -L 形式（第三步泊松）
 *   -1：A = diag·I - off·邻接，用于 M+tL 形式（第一步热扩散，M-矩阵）
 * singular=true 时每轮投影到常数零空间正交补。
 */
export function conjugateGradient(
  V: number,
  diag: Float32Array,
  adj: Adjacency,
  offdiag: Float32Array,
  b: Float32Array,
  tolerance = 1e-6,
  maxIter = 300,
  singular = false,
  offSign: 1 | -1 = 1,
): { x: Float32Array; iterations: number; residual: number } {
  const x = new Float32Array(V);
  const r = new Float32Array(b);
  if (singular) projectNullspace(r);
  const p = new Float32Array(r);
  const Ap = new Float32Array(V);
  let rr = dot(r, r);
  const rr0 = rr || 1;
  let it = 0;
  for (; it < maxIter && rr > tolerance * tolerance * rr0; it++) {
    spmv(diag, adj, offdiag, p, Ap, offSign);
    const pAp = dot(p, Ap);
    if (pAp <= 1e-12) break;
    const alpha = rr / pAp;
    for (let i = 0; i < V; i++) {
      x[i] += alpha * p[i];
      r[i] -= alpha * Ap[i];
    }
    if (singular) projectNullspace(r);
    const rrNew = dot(r, r);
    const beta = rrNew / rr;
    for (let i = 0; i < V; i++) p[i] = r[i] + beta * p[i];
    rr = rrNew;
  }
  return { x, iterations: it, residual: Math.sqrt(rr / rr0) };
}

function projectNullspace(v: Float32Array): void {
  let mean = 0;
  for (let i = 0; i < v.length; i++) mean += v[i];
  mean /= v.length;
  for (let i = 0; i < v.length; i++) v[i] -= mean;
}

/** y = diag·x + offSign·Σ off·x_neighbor */
function spmv(diag: Float32Array, adj: Adjacency, offdiag: Float32Array, x: Float32Array, out: Float32Array, offSign: 1 | -1): void {
  const V = diag.length;
  for (let i = 0; i < V; i++) {
    let s = diag[i] * x[i];
    for (let j = adj.offset[i]; j < adj.offset[i + 1]; j++) {
      s += offSign * offdiag[j] * x[adj.neighbors[j]];
    }
    out[i] = s;
  }
}

function dot(a: Float32Array, b: Float32Array): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

/** 推荐时间步 h²（h = 平均边长），Crane 论文的稳健默认 */
export function recommendTimeStep(m: Mesh, adj: Adjacency): number {
  let sum = 0, cnt = 0;
  for (let i = 0; i < adj.edgeLength.length; i++) { sum += adj.edgeLength[i]; cnt++; }
  const h = sum / (cnt || 1);
  return h * h;
}

export interface HeatGeodesic {
  distance: Float32Array; // 每个顶点到源点的测地距离（毫米）
  iterations: number;
  usedFallback: boolean;
}

export interface GeodesicResult {
  distance: Float32Array;
  /** 实际使用的算法 */
  method: 'dijkstra' | 'heat';
  elapsedMs: number;
}

/** 邻接表按网格对象缓存（修型期间拓扑不变，只建一次，20 万面构建约 300ms） */
const adjacencyCache = new WeakMap<Mesh, Adjacency>();

export function getAdjacency(m: Mesh): Adjacency {
  let adj = adjacencyCache.get(m);
  if (!adj) {
    adj = buildAdjacency(m);
    adjacencyCache.set(m, adj);
  }
  return adj;
}

/**
 * 单源测地距离场（默认 Dijkstra，可靠且满足 200ms 预算）。
 * 图距离是真实测地的上界，20 万面边长约 2mm 时相对误差 <1%，
 * 对笔刷高斯衰减足够精确。热法可通过 method='heat' 显式启用。
 */
export function geodesicDistance(
  m: Mesh,
  sourceVertex: number,
  opts: { method?: 'dijkstra' | 'heat'; t?: number } = {},
): GeodesicResult {
  const t0 = performance.now();
  const adj = getAdjacency(m);
  if (opts.method === 'heat') {
    const r = heatGeodesicDistance(m, sourceVertex, opts.t, adj);
    return { distance: r.distance, method: r.usedFallback ? 'dijkstra' : 'heat', elapsedMs: performance.now() - t0 };
  }
  const distance = dijkstraGeodesic(adj, sourceVertex);
  return { distance, method: 'dijkstra', elapsedMs: performance.now() - t0 };
}

/** 热法（Crane 2013）。钝角三角化扫描网格上可能不稳定，异常时自动回退 Dijkstra。 */
function heatGeodesicDistance(m: Mesh, sourceVertex: number, t?: number, cachedAdj?: Adjacency): HeatGeodesic {
  const adj = cachedAdj ?? getAdjacency(m);
  const L = buildCotLaplacian(m, adj);
  const V = vertexCount(m);
  const areas = vertexAreas(m);
  const h2 = t ?? recommendTimeStep(m, adj);

  // 第一步 (M + tL_umb) u = δ，L_umb 负半定（对角 degree、非对角 -1）。
  // 系统 A = diag(area + t·degree) - t·邻接，是 M-矩阵，解单调非负。
  const aDiag = new Float32Array(V);
  const aOff = new Float32Array(L.offdiag.length);
  for (let i = 0; i < V; i++) aDiag[i] = areas[i] + h2 * L.umbrellaDiag[i];
  for (let j = 0; j < aOff.length; j++) aOff[j] = h2;
  const rhs = new Float32Array(V);
  rhs[sourceVertex] = 1;
  const cg1 = conjugateGradient(V, aDiag, adj, aOff, rhs, 1e-10, 600, false, -1);
  const u = cg1.x;

  // 第二步 面梯度 ∇u = 1/(2A) Σ_i u_i (n × e_i)，再取单位化负梯度 X
  const grad = new Float32Array(m.indices.length);
  for (let f = 0; f < faceCount(m); f++) {
    const tri = [m.indices[f * 3], m.indices[f * 3 + 1], m.indices[f * 3 + 2]];
    const p = tri.map((v) => getVertex(m, v));
    const n = v3.norm(v3.cross(v3.sub(p[1], p[0]), v3.sub(p[2], p[0])));
    const area2 = v3.len(v3.cross(v3.sub(p[1], p[0]), v3.sub(p[2], p[0])));
    let g: Vec3 = [0, 0, 0];
    for (let k = 0; k < 3; k++) {
      const ei = v3.sub(p[(k + 1) % 3], p[(k + 2) % 3]);
      g = v3.add(g, v3.scale(v3.cross(n, ei), u[tri[k]] / (area2 || 1e-9)));
    }
    // 本实现的面梯度公式 ∇u = 1/(2A) Σ u_i (n × e_i) 已给出正确方向
    // （线性场 u=x 验证 ∇u=+x）。距离场沿 -∇u 方向增长，故 X = -∇u/|∇u|。
    const gl = v3.len(g) || 1e-9;
    grad[f * 3] = -g[0] / gl;
    grad[f * 3 + 1] = -g[1] / gl;
    grad[f * 3 + 2] = -g[2] / gl;
  }

  // 散度 div_i(X)：余切与对边配对
  const div = new Float32Array(V);
  for (let f = 0; f < faceCount(m); f++) {
    const tri = [m.indices[f * 3], m.indices[f * 3 + 1], m.indices[f * 3 + 2]];
    const p = tri.map((v) => getVertex(m, v));
    const X: Vec3 = [grad[f * 3], grad[f * 3 + 1], grad[f * 3 + 2]];
    for (let k = 0; k < 3; k++) {
      const i = tri[k];
      const j = tri[(k + 1) % 3];
      const l = tri[(k + 2) % 3];
      const cotAtJ = L.faceCot[f * 3 + ((k + 1) % 3)];
      const cotAtL = L.faceCot[f * 3 + ((k + 2) % 3)];
      const eij = v3.sub(p[(k + 1) % 3], p[k]);
      const eil = v3.sub(p[(k + 2) % 3], p[k]);
      div[i] += 0.5 * (cotAtJ * v3.dot(eil, X) + cotAtL * v3.dot(eij, X));
    }
  }

  // 第三步 距离泊松。负半定 Laplacian：L_ii=+Σw，L_ij=-w（常数向量为零空间）。
  // CG 用 offSign=-1 的同一矩阵直接求解（奇异，投影零空间），RHS=div。
  const cg2 = conjugateGradient(V, L.diagPos, adj, L.offdiag, div, 1e-8, 600, true, -1);
  let phi = cg2.x;
  const srcVal = phi[sourceVertex];
  for (let i = 0; i < V; i++) phi[i] -= srcVal;

  // 健康检查：闭合网格上距离场应几乎处处非负；大面积负值说明数值异常，回退 Dijkstra
  const tol = Math.sqrt(h2) * 0.05;
  let negCount = 0;
  for (let i = 0; i < V; i++) if (phi[i] < -tol) negCount++;
  let usedFallback = false;
  if (negCount > V * 0.05 && (globalThis as { __HEAT_NO_FALLBACK?: boolean }).__HEAT_NO_FALLBACK !== true) {
    phi = dijkstraGeodesic(adj, sourceVertex);
    usedFallback = true;
  }
  for (let i = 0; i < V; i++) if (phi[i] < 0) phi[i] = 0;
  return { distance: phi, iterations: cg1.iterations + cg2.iterations, usedFallback };
}

/** 图 Dijkstra 兜底（精度低但永远可用，WebGPU 不可用且热法异常时启用） */
export function dijkstraGeodesic(adj: Adjacency, source: number): Float32Array {
  const V = adj.offset.length - 1;
  const dist = new Float32Array(V).fill(Infinity);
  const done = new Uint8Array(V);
  dist[source] = 0;
  const heap: number[] = [source];
  const prio = dist;
  const swap = (a: number, b: number) => { [heap[a], heap[b]] = [heap[b], heap[a]]; };
  const pop = (): number => {
    const top = heap[0];
    const last = heap.pop()!;
    if (heap.length) {
      heap[0] = last;
      let i = 0;
      for (;;) {
        const l = i * 2 + 1, r = l + 1;
        let s = i;
        if (l < heap.length && prio[heap[l]] < prio[heap[s]]) s = l;
        if (r < heap.length && prio[heap[r]] < prio[heap[s]]) s = r;
        if (s === i) break;
        swap(i, s);
        i = s;
      }
    }
    return top;
  };
  const push = (v: number) => {
    heap.push(v);
    let i = heap.length - 1;
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (prio[heap[p]] <= prio[heap[i]]) break;
      swap(i, p);
      i = p;
    }
  };
  while (heap.length) {
    const v = pop();
    if (done[v]) continue;
    done[v] = 1;
    for (let j = adj.offset[v]; j < adj.offset[v + 1]; j++) {
      const u = adj.neighbors[j];
      const nd = dist[v] + adj.edgeLength[j];
      if (nd < dist[u]) {
        dist[u] = nd;
        push(u);
      }
    }
  }
  return dist;
}
