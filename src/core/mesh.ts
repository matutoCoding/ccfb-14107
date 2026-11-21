// 三角网格核心数据结构：位置/索引 + 完整拓扑邻接 + 测地/面积/法向等几何量
import {
  Vec3,
  add,
  bounds,
  cross,
  dot,
  length,
  normalize,
  scale,
  sub,
} from './math';

export interface MeshData {
  positions: Float32Array;
  indices: Uint32Array;
  /** 可选顶点色（RGB，0..1），用于色谱渲染 */
  colors?: Float32Array;
}

export class TriangleMesh {
  positions: Float32Array;
  indices: Uint32Array;
  colors: Float32Array | null;

  // —— 拓扑（惰性构建）——
  private _v2v: Uint32Array[] | null = null;
  private _v2f: Uint32Array[] | null = null;
  private _faceAdj: Int32Array | null = null; // 每面 3 个邻面，-1 为边界
  private _edges: Uint32Array | null = null; // 去重边，每条 2 个顶点
  private _boundaryVerts: Uint32Array | null = null;

  constructor(positions: Float32Array, indices: Uint32Array, colors?: Float32Array | null) {
    if (indices.length % 3 !== 0) throw new Error('索引数必须是 3 的倍数');
    this.positions = positions;
    this.indices = indices;
    this.colors = colors ?? null;
  }

  get vertexCount(): number {
    return this.positions.length / 3;
  }
  get faceCount(): number {
    return this.indices.length / 3;
  }

  vertex(i: number): Vec3 {
    return [this.positions[i * 3], this.positions[i * 3 + 1], this.positions[i * 3 + 2]];
  }
  face(f: number): [number, number, number] {
    const o = f * 3;
    return [this.indices[o], this.indices[o + 1], this.indices[o + 2]];
  }

  /** 顶点-顶点邻接表（无重复） */
  v2v(): Uint32Array[] {
    if (this._v2v) return this._v2v;
    const sets: Set<number>[] = Array.from({ length: this.vertexCount }, () => new Set());
    const { indices } = this;
    for (let i = 0; i < indices.length; i += 3) {
      const a = indices[i],
        b = indices[i + 1],
        c = indices[i + 2];
      sets[a].add(b); sets[a].add(c);
      sets[b].add(a); sets[b].add(c);
      sets[c].add(a); sets[c].add(b);
    }
    this._v2v = sets.map((s) => Uint32Array.from(s));
    return this._v2v;
  }

  /** 顶点-面邻接表 */
  v2f(): Uint32Array[] {
    if (this._v2f) return this._v2f;
    const lists: number[][] = Array.from({ length: this.vertexCount }, () => []);
    for (let f = 0; f < this.faceCount; f++) {
      const o = f * 3;
      lists[this.indices[o]].push(f);
      lists[this.indices[o + 1]].push(f);
      lists[this.indices[o + 2]].push(f);
    }
    this._v2f = lists.map((l) => Uint32Array.from(l));
    return this._v2f;
  }

  /** 面邻接：faceAdj[f*3+k] 为与第 k 条对边相邻的面，边界为 -1 */
  faceAdjacency(): Int32Array {
    if (this._faceAdj) return this._faceAdj;
    const adj = new Int32Array(this.faceCount * 3).fill(-1);
    const map = new Map<number, number>(); // 边 hash -> (face<<2 | slot)
    const edgeKey = (a: number, b: number) => {
      const lo = Math.min(a, b),
        hi = Math.max(a, b);
      return lo * this.vertexCount + hi;
    };
    for (let f = 0; f < this.faceCount; f++) {
      const [a, b, c] = this.face(f);
      const edges: Array<[number, number]> = [
        [b, c],
        [c, a],
        [a, b],
      ];
      for (let k = 0; k < 3; k++) {
        const key = edgeKey(edges[k][0], edges[k][1]);
        const prev = map.get(key);
        if (prev === undefined) {
          map.set(key, (f << 2) | k);
        } else {
          const pf = prev >>> 2;
          const pk = prev & 3;
          adj[f * 3 + k] = pf;
          adj[pf * 3 + pk] = f;
        }
      }
    }
    this._faceAdj = adj;
    return adj;
  }

  /** 去重有向边表（每两条 uint32 一条边） */
  edges(): Uint32Array {
    if (this._edges) return this._edges;
    const set = new Set<number>();
    const key = (a: number, b: number) => {
      const lo = Math.min(a, b),
        hi = Math.max(a, b);
      return lo * this.vertexCount + hi;
    };
    const out: number[] = [];
    for (let i = 0; i < this.indices.length; i += 3) {
      const tri = [this.indices[i], this.indices[i + 1], this.indices[i + 2]];
      for (let k = 0; k < 3; k++) {
        const a = tri[k],
          b = tri[(k + 1) % 3];
        const h = key(a, b);
        if (!set.has(h)) {
          set.add(h);
          out.push(a, b);
        }
      }
    }
    this._edges = Uint32Array.from(out);
    return this._edges;
  }

  /** 边界顶点集合（只出现在一条面里的边的端点） */
  boundaryVertices(): Uint32Array {
    if (this._boundaryVerts) return this._boundaryVerts;
    const adj = this.faceAdjacency();
    const set = new Set<number>();
    for (let f = 0; f < this.faceCount; f++) {
      for (let k = 0; k < 3; k++) {
        if (adj[f * 3 + k] === -1) {
          const [a, b, c] = this.face(f);
          const tri = [a, b, c];
          set.add(tri[k]);
          set.add(tri[(k + 1) % 3]);
        }
      }
    }
    this._boundaryVerts = Uint32Array.from(set);
    return this._boundaryVerts;
  }

  /** 每个三角形的面积 */
  faceAreas(): Float32Array {
    const areas = new Float32Array(this.faceCount);
    const p = this.positions,
      idx = this.indices;
    for (let f = 0; f < this.faceCount; f++) {
      const o = f * 3;
      const a = idx[o] * 3,
        b = idx[o + 1] * 3,
        c = idx[o + 2] * 3;
      const e1: Vec3 = [p[b] - p[a], p[b + 1] - p[a + 1], p[b + 2] - p[a + 2]];
      const e2: Vec3 = [p[c] - p[a], p[c + 1] - p[a + 1], p[c + 2] - p[a + 2]];
      areas[f] = 0.5 * length(cross(e1, e2));
    }
    return areas;
  }

  /** 顶点混合面积（Voronoi 面积近似），热方程质量矩阵用 */
  vertexAreas(): Float32Array {
    const areas = new Float32Array(this.vertexCount);
    const fa = this.faceAreas();
    for (let f = 0; f < this.faceCount; f++) {
      const third = fa[f] / 3;
      const o = f * 3;
      areas[this.indices[o]] += third;
      areas[this.indices[o + 1]] += third;
      areas[this.indices[o + 2]] += third;
    }
    return areas;
  }

  /** 面积加权顶点法向（朝外） */
  vertexNormals(): Float32Array {
    const n = new Float32Array(this.vertexCount * 3);
    const p = this.positions,
      idx = this.indices;
    for (let f = 0; f < this.faceCount; f++) {
      const o = f * 3;
      const ia = idx[o],
        ib = idx[o + 1],
        ic = idx[o + 2];
      const a = ia * 3,
        b = ib * 3,
        c = ic * 3;
      const e1: Vec3 = [p[b] - p[a], p[b + 1] - p[a + 1], p[b + 2] - p[a + 2]];
      const e2: Vec3 = [p[c] - p[a], p[c + 1] - p[a + 1], p[c + 2] - p[a + 2]];
      const fn = cross(e1, e2); // 含面积权重
      n[a] += fn[0]; n[a + 1] += fn[1]; n[a + 2] += fn[2];
      n[b] += fn[0]; n[b + 1] += fn[1]; n[b + 2] += fn[2];
      n[c] += fn[0]; n[c + 1] += fn[1]; n[c + 2] += fn[2];
    }
    for (let i = 0; i < this.vertexCount; i++) {
      const l = Math.hypot(n[i * 3], n[i * 3 + 1], n[i * 3 + 2]) || 1;
      n[i * 3] /= l;
      n[i * 3 + 1] /= l;
      n[i * 3 + 2] /= l;
    }
    return n;
  }

  /**
   * 余切权重（拉普拉斯矩阵 L 的非对角元 -w_ij），CSR 三数组。
   * 对角为同行和，边界/退化三角形做钳制，保证半正定。
   */
  cotanLaplacianCSR(): { offsets: Uint32Array; indices: Uint32Array; values: Float32Array } {
    const n = this.vertexCount;
    const w = new Map<number, number>();
    const wKey = (i: number, j: number) => i * n + j;
    const p = this.positions,
      idx = this.indices;
    for (let f = 0; f < this.faceCount; f++) {
      const o = f * 3;
      const ia = idx[o],
        ib = idx[o + 1],
        ic = idx[o + 2];
      const pa = this.vertex(ia),
        pb = this.vertex(ib),
        pc = this.vertex(ic);
      const cot = (opp: Vec3, q1: Vec3, q2: Vec3): number => {
        const u = sub(q1, opp);
        const v = sub(q2, opp);
        const d = length(cross(u, v));
        if (d < 1e-12) return 0;
        return Math.max(1e-6, dot(u, v) / d); // 钳制避免负权
      };
      const pairs: Array<[number, number, number]> = [
        [ia, ib, cot(pc, pa, pb)],
        [ib, ic, cot(pa, pb, pc)],
        [ic, ia, cot(pb, pc, pa)],
      ];
      for (const [i, j, cw] of pairs) {
        const half = 0.5 * cw;
        w.set(wKey(i, j), (w.get(wKey(i, j)) ?? 0) + half);
        w.set(wKey(j, i), (w.get(wKey(j, i)) ?? 0) + half);
      }
    }
    // 组装 CSR（含对角）
    const offsets = new Uint32Array(n + 1);
    const neighborSets: number[][] = Array.from({ length: n }, () => []);
    for (const key of w.keys()) {
      const i = Math.floor(key / n);
      neighborSets[i].push(key % n);
    }
    const entries: Array<{ j: number; v: number }[]> = neighborSets.map((ns, i) => {
      ns.sort((a, b) => a - b);
      const row = ns.map((j) => ({ j, v: -(w.get(wKey(i, j)) ?? 0) }));
      const diag = row.reduce((s, e) => s - e.v, 0);
      row.push({ j: i, v: diag });
      row.sort((a, b) => a.j - b.j);
      return row;
    });
    let count = 0;
    for (let i = 0; i < n; i++) {
      offsets[i] = count;
      count += entries[i].length;
    }
    offsets[n] = count;
    const csrIdx = new Uint32Array(count);
    const csrVal = new Float32Array(count);
    for (let i = 0; i < n; i++) {
      for (let k = 0; k < entries[i].length; k++) {
        const e = entries[i][k];
        csrIdx[offsets[i] + k] = e.j;
        csrVal[offsets[i] + k] = e.v;
      }
    }
    return { offsets, indices: csrIdx, values: csrVal };
  }

  /**
   * CPU Dijkstra 精确测地距离（边长图距离，热法的对照基准与测试用）。
   * sources 可带初始距离。返回每顶点距离。
   */
  geodesicDijkstra(sources: Map<number, number>): Float32Array {
    const n = this.vertexCount;
    const dist = new Float32Array(n).fill(Infinity);
    const visited = new Uint8Array(n);
    const adj = this.v2v();
    const p = this.positions;
    // 简单二叉堆
    const heap: number[] = [];
    const prio = new Float32Array(n).fill(Infinity);
    const swap = (a: number, b: number) => {
      const t = heap[a];
      heap[a] = heap[b];
      heap[b] = t;
    };
    const push = (v: number, d: number) => {
      prio[v] = d;
      heap.push(v);
      let i = heap.length - 1;
      while (i > 0) {
        const parent = (i - 1) >> 1;
        if (prio[heap[parent]] <= prio[heap[i]]) break;
        swap(i, parent);
        i = parent;
      }
    };
    const pop = (): number => {
      const top = heap[0];
      const last = heap.pop()!;
      if (heap.length > 0) {
        heap[0] = last;
        let i = 0;
        for (;;) {
          const l = i * 2 + 1,
            r = l + 1;
          let smallest = i;
          if (l < heap.length && prio[heap[l]] < prio[heap[smallest]]) smallest = l;
          if (r < heap.length && prio[heap[r]] < prio[heap[smallest]]) smallest = r;
          if (smallest === i) break;
          swap(i, smallest);
          i = smallest;
        }
      }
      return top;
    };
    for (const [s, d0] of sources) {
      dist[s] = d0;
      push(s, d0);
    }
    while (heap.length) {
      const u = pop();
      if (visited[u]) continue;
      visited[u] = 1;
      const du = dist[u];
      const ux = p[u * 3],
        uy = p[u * 3 + 1],
        uz = p[u * 3 + 2];
      for (const v of adj[u]) {
        if (visited[v]) continue;
        const dx = p[v * 3] - ux,
          dy = p[v * 3 + 1] - uy,
          dz = p[v * 3 + 2] - uz;
        const nd = du + Math.hypot(dx, dy, dz);
        if (nd < dist[v]) {
          dist[v] = nd;
          push(v, nd);
        }
      }
    }
    return dist;
  }

  /** 均匀空间哈希：点/面近邻查询，CPU 端评估与导出校验用 */
  spatialHash(cellSize: number): SpatialHash {
    return new SpatialHash(this, cellSize);
  }

  bbox() {
    return bounds(this.positions);
  }

  /** 应用 4x4 变换（列主序，齐次 w=1），返回新网格 */
  transformed(m: Float32Array): TriangleMesh {
    const out = new Float32Array(this.positions.length);
    const p = this.positions;
    for (let i = 0; i < this.vertexCount; i++) {
      const x = p[i * 3],
        y = p[i * 3 + 1],
        z = p[i * 3 + 2];
      out[i * 3] = m[0] * x + m[4] * y + m[8] * z + m[12];
      out[i * 3 + 1] = m[1] * x + m[5] * y + m[9] * z + m[13];
      out[i * 3 + 2] = m[2] * x + m[6] * y + m[10] * z + m[14];
    }
    return new TriangleMesh(out, this.indices, this.colors);
  }

  clone(): TriangleMesh {
    return new TriangleMesh(
      new Float32Array(this.positions),
      new Uint32Array(this.indices),
      this.colors ? new Float32Array(this.colors) : null,
    );
  }

  /** 统计欧拉特征，供水密性判断 */
  eulerCharacteristic(): number {
    return this.vertexCount - this.edges().length / 2 + this.faceCount;
  }

  isWatertight(): boolean {
    return this.boundaryVertices().length === 0;
  }
}

/** 均匀网格空间哈希（按面 AABB 入桶） */
export class SpatialHash {
  readonly cellSize: number;
  private buckets = new Map<number, number[]>();
  private mesh: TriangleMesh;

  constructor(mesh: TriangleMesh, cellSize: number) {
    this.mesh = mesh;
    this.cellSize = cellSize;
    const p = mesh.positions,
      idx = mesh.indices;
    for (let f = 0; f < mesh.faceCount; f++) {
      const o = f * 3;
      let minX = Infinity, minY = Infinity, minZ = Infinity;
      let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
      for (let k = 0; k < 3; k++) {
        const vi = idx[o + k] * 3;
        minX = Math.min(minX, p[vi]); maxX = Math.max(maxX, p[vi]);
        minY = Math.min(minY, p[vi + 1]); maxY = Math.max(maxY, p[vi + 1]);
        minZ = Math.min(minZ, p[vi + 2]); maxZ = Math.max(maxZ, p[vi + 2]);
      }
      const i0 = Math.floor(minX / cellSize), i1 = Math.floor(maxX / cellSize);
      const j0 = Math.floor(minY / cellSize), j1 = Math.floor(maxY / cellSize);
      const k0 = Math.floor(minZ / cellSize), k1 = Math.floor(maxZ / cellSize);
      for (let i = i0; i <= i1; i++)
        for (let j = j0; j <= j1; j++)
          for (let k = k0; k <= k1; k++) {
            const h = this.hash(i, j, k);
            let arr = this.buckets.get(h);
            if (!arr) this.buckets.set(h, (arr = []));
            arr.push(f);
          }
    }
  }

  private hash(i: number, j: number, k: number): number {
    // 大质数混合，避免负坐标冲突
    return ((i * 73856093) ^ (j * 19349663) ^ (k * 83492791)) | 0;
  }

  /** 候选面：点周围半径 r 内的桶 */
  candidateFaces(x: number, y: number, z: number, r: number): number[] {
    const i0 = Math.floor((x - r) / this.cellSize), i1 = Math.floor((x + r) / this.cellSize);
    const j0 = Math.floor((y - r) / this.cellSize), j1 = Math.floor((y + r) / this.cellSize);
    const k0 = Math.floor((z - r) / this.cellSize), k1 = Math.floor((z + r) / this.cellSize);
    const seen = new Set<number>();
    const out: number[] = [];
    for (let i = i0; i <= i1; i++)
      for (let j = j0; j <= j1; j++)
        for (let k = k0; k <= k1; k++) {
          const arr = this.buckets.get(this.hash(i, j, k));
          if (arr) for (const f of arr) if (!seen.has(f)) { seen.add(f); out.push(f); }
        }
    return out;
  }

  /** 点到网格最近三角面的距离与命中信息 */
  nearest(x: number, y: number, z: number, searchRadius = this.cellSize * 4): {
    distance: number;
    face: number;
    point: Vec3;
  } | null {
    let radius = this.cellSize;
    let best: { distance: number; face: number; point: Vec3 } | null = null;
    for (let attempt = 0; attempt < 6; attempt++) {
      const cands = this.candidateFaces(x, y, z, radius);
      for (const f of cands) {
        const hit = pointTriangleDistance(this.mesh, f, [x, y, z]);
        if (!best || hit.distance < best.distance) best = { ...hit, face: f };
      }
      if (best && best.distance <= radius) break;
      radius *= 2;
      if (radius > searchRadius) break;
    }
    return best;
  }
}

/** 点到三角形最近距离（Ericson Real-Time Collision Detection 算法） */
export function pointTriangleDistance(
  mesh: TriangleMesh,
  face: number,
  q: Vec3,
): { distance: number; point: Vec3 } {
  const [a, b, c] = mesh.face(face).map((v) => mesh.vertex(v)) as [Vec3, Vec3, Vec3];
  const ab = sub(b, a);
  const ac = sub(c, a);
  const ap = sub(q, a);
  const d1 = dot(ab, ap);
  const d2 = dot(ac, ap);
  if (d1 <= 0 && d2 <= 0) return { distance: length(sub(q, a)), point: a };
  const bp = sub(q, b);
  const d3 = dot(ab, bp);
  const d4 = dot(ac, bp);
  if (d3 >= 0 && d4 <= d3) return { distance: length(sub(q, b)), point: b };
  const vc = d1 * d4 - d3 * d2;
  if (vc <= 0 && d1 >= 0 && d3 <= 0) {
    const v = d1 / (d1 - d3);
    const pt = add(a, scale(ab, v));
    return { distance: length(sub(q, pt)), point: pt };
  }
  const cp = sub(q, c);
  const d5 = dot(ab, cp);
  const d6 = dot(ac, cp);
  if (d6 >= 0 && d5 <= d6) return { distance: length(sub(q, c)), point: c };
  const vb = d5 * d2 - d1 * d6;
  if (vb <= 0 && d2 >= 0 && d6 <= 0) {
    const w = d2 / (d2 - d6);
    const pt = add(a, scale(ac, w));
    return { distance: length(sub(q, pt)), point: pt };
  }
  const va = d3 * d6 - d5 * d4;
  if (va <= 0 && d4 - d3 >= 0 && d5 - d6 >= 0) {
    const w = (d4 - d3) / (d4 - d3 + (d5 - d6));
    const pt = add(b, scale(sub(c, b), w));
    return { distance: length(sub(q, pt)), point: pt };
  }
  const denom = 1 / (va + vb + vc);
  const v = vb * denom;
  const w = vc * denom;
  const pt = add(a, add(scale(ab, v), scale(ac, w)));
  return { distance: length(sub(q, pt)), point: pt };
}

/** 射线与三角形求交（拾取用），返回 t 与重心坐标 */
export function rayTriangle(
  mesh: TriangleMesh,
  face: number,
  origin: Vec3,
  dir: Vec3,
): { t: number; u: number; v: number } | null {
  const [ia, ib, ic] = mesh.face(face);
  const a = mesh.vertex(ia),
    b = mesh.vertex(ib),
    c = mesh.vertex(ic);
  const e1 = sub(b, a);
  const e2 = sub(c, a);
  const p = cross(dir, e2);
  const det = dot(e1, p);
  if (Math.abs(det) < 1e-10) return null;
  const invDet = 1 / det;
  const tvec = sub(origin, a);
  const u = dot(tvec, p) * invDet;
  if (u < 0 || u > 1) return null;
  const qv = cross(tvec, e1);
  const v = dot(dir, qv) * invDet;
  if (v < 0 || u + v > 1) return null;
  const t = dot(e2, qv) * invDet;
  return t > 0 ? { t, u, v } : null;
}

/** 构建拓扑闭合的 UV 球（单极点 + 环向首尾相接，无接缝重复顶点），测试用 */
export function makeSphereMesh(segments = 48, rings = 32, radius = 1): TriangleMesh {
  const pos: number[] = [];
  const idx: number[] = [];
  const north = pos.length / 3;
  pos.push(0, radius, 0);
  // 中间环：rings-1 行，每行 segments 个顶点
  const ringStart = (r: number) => 1 + (r - 1) * segments;
  for (let r = 1; r < rings; r++) {
    const theta = (r / rings) * Math.PI;
    for (let s = 0; s < segments; s++) {
      const phi = (s / segments) * Math.PI * 2;
      pos.push(
        radius * Math.sin(theta) * Math.cos(phi),
        radius * Math.cos(theta),
        radius * Math.sin(theta) * Math.sin(phi),
      );
    }
  }
  const south = pos.length / 3;
  pos.push(0, -radius, 0);
  const wrap = (s: number) => (s + segments) % segments;
  // 北极扇
  for (let s = 0; s < segments; s++) {
    idx.push(north, ringStart(1) + wrap(s + 1), ringStart(1) + wrap(s));
  }
  // 环间四边形
  for (let r = 1; r < rings - 1; r++) {
    const cur = ringStart(r);
    const nxt = ringStart(r + 1);
    for (let s = 0; s < segments; s++) {
      const a = cur + wrap(s);
      const b = nxt + wrap(s);
      const c = cur + wrap(s + 1);
      const d = nxt + wrap(s + 1);
      idx.push(a, c, b, b, c, d);
    }
  }
  // 南极扇
  const lastRing = ringStart(rings - 1);
  for (let s = 0; s < segments; s++) {
    idx.push(south, lastRing + wrap(s), lastRing + wrap(s + 1));
  }
  return new TriangleMesh(Float32Array.from(pos), Uint32Array.from(idx));
}

export function normalizeMeshNormals(mesh: TriangleMesh): void {
  // 基于质心-顶点方向的全局法向一致性修正（躯干为闭合/近闭合网格）
  const n = mesh.vertexNormals();
  const { center } = mesh.bbox();
  const p = mesh.positions;
  let outward = 0;
  for (let i = 0; i < mesh.vertexCount; i += Math.max(1, Math.floor(mesh.vertexCount / 500))) {
    const v: Vec3 = [p[i * 3] - center[0], p[i * 3 + 1] - center[1], p[i * 3 + 2] - center[2]];
    outward += dot(v, [n[i * 3], n[i * 3 + 1], n[i * 3 + 2]]);
  }
  if (outward < 0) {
    // 翻转绕序
    for (let i = 0; i < mesh.indices.length; i += 3) {
      const t = mesh.indices[i + 1];
      mesh.indices[i + 1] = mesh.indices[i + 2];
      mesh.indices[i + 2] = t;
    }
  }
}
