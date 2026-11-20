import { Vec3, v3 } from './math';

/**
 * 三角网格。单位：毫米。
 * positions 长度 = 顶点数 * 3，indices 长度 = 面数 * 3。
 */
export interface Mesh {
  positions: Float32Array;
  indices: Uint32Array;
  /** 可选顶点色（RGB，0..1），长度 = 顶点数 * 3 */
  colors?: Float32Array;
  name?: string;
}

export function vertexCount(m: Mesh): number {
  return m.positions.length / 3;
}

export function faceCount(m: Mesh): number {
  return m.indices.length / 3;
}

export function getVertex(m: Mesh, i: number): Vec3 {
  return [m.positions[i * 3], m.positions[i * 3 + 1], m.positions[i * 3 + 2]];
}

export function setVertex(m: Mesh, i: number, p: Vec3): void {
  m.positions[i * 3] = p[0];
  m.positions[i * 3 + 1] = p[1];
  m.positions[i * 3 + 2] = p[2];
}

export function cloneMesh(m: Mesh): Mesh {
  return {
    positions: new Float32Array(m.positions),
    indices: new Uint32Array(m.indices),
    colors: m.colors ? new Float32Array(m.colors) : undefined,
    name: m.name,
  };
}

export function meshBounds(m: Mesh): { min: Vec3; max: Vec3; center: Vec3; size: Vec3 } {
  const min: Vec3 = [Infinity, Infinity, Infinity];
  const max: Vec3 = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < m.positions.length; i += 3) {
    for (let k = 0; k < 3; k++) {
      min[k] = Math.min(min[k], m.positions[i + k]);
      max[k] = Math.max(max[k], m.positions[i + k]);
    }
  }
  return {
    min,
    max,
    center: v3.scale(v3.add(min, max), 0.5),
    size: v3.sub(max, min),
  };
}

/** CSR 形式的顶点邻接表：顶点 i 的邻居在 nbr[offset[i]..offset[i+1]) */
export interface Adjacency {
  offset: Uint32Array; // 长度 V+1
  neighbors: Uint32Array;
  /** 与 neighbors 对齐：共享边的边长（毫米） */
  edgeLength: Float32Array;
  /** 每个顶点的关联面（同样 CSR，布局与 offset 一致） */
  faceOffset: Uint32Array;
  vertexFaces: Uint32Array;
}

/** 构建顶点邻接与顶点-面关联（扫描/修型全程复用，只建一次） */
export function buildAdjacency(m: Mesh): Adjacency {
  const V = vertexCount(m);
  const F = faceCount(m);
  const faceDegree = new Uint32Array(V);
  // 真实邻接度数：用无向边集合去重（闭合网格共享边出现两次，不能按半边计数）
  const edgeSet = new Set<string>();
  const ekey = (a: number, b: number) => (a < b ? `${a}|${b}` : `${b}|${a}`);
  for (let f = 0; f < F; f++) {
    const a = m.indices[f * 3];
    const b = m.indices[f * 3 + 1];
    const c = m.indices[f * 3 + 2];
    faceDegree[a]++; faceDegree[b]++; faceDegree[c]++;
    edgeSet.add(ekey(a, b));
    edgeSet.add(ekey(b, c));
    edgeSet.add(ekey(c, a));
  }
  const degree = new Uint32Array(V);
  for (const key of edgeSet) {
    const [a, b] = key.split('|');
    degree[+a]++;
    degree[+b]++;
  }
  const offset = new Uint32Array(V + 1);
  const faceOffset = new Uint32Array(V + 1);
  for (let i = 0; i < V; i++) {
    offset[i + 1] = offset[i] + degree[i];
    faceOffset[i + 1] = faceOffset[i] + faceDegree[i];
  }
  const neighbors = new Uint32Array(offset[V]);
  const edgeLength = new Float32Array(offset[V]);
  const vertexFaces = new Uint32Array(faceOffset[V]);
  const cursor = new Uint32Array(V);
  const faceCursor = new Uint32Array(V);
  for (let f = 0; f < F; f++) {
    const tri = [m.indices[f * 3], m.indices[f * 3 + 1], m.indices[f * 3 + 2]];
    for (let k = 0; k < 3; k++) {
      const v = tri[k];
      vertexFaces[faceOffset[v] + faceCursor[v]++] = f;
      // 每个顶点各自去重：无向边要同时写进两端的邻接表
      const localSeen = new Set<number>();
      for (let j = offset[v]; j < offset[v] + cursor[v]; j++) localSeen.add(neighbors[j]);
      for (const u of [tri[(k + 1) % 3], tri[(k + 2) % 3]]) {
        if (localSeen.has(u)) continue;
        localSeen.add(u);
        const pos = offset[v] + cursor[v]++;
        neighbors[pos] = u;
        edgeLength[pos] = v3.dist(getVertex(m, v), getVertex(m, u));
      }
    }
  }
  return {
    offset,
    neighbors,
    edgeLength,
    faceOffset,
    vertexFaces,
  };
}

/** 面法线（单位化） */
export function faceNormals(m: Mesh): Float32Array {
  const F = faceCount(m);
  const out = new Float32Array(F * 3);
  for (let f = 0; f < F; f++) {
    const a = m.indices[f * 3] * 3;
    const b = m.indices[f * 3 + 1] * 3;
    const c = m.indices[f * 3 + 2] * 3;
    const ab: Vec3 = [
      m.positions[b] - m.positions[a],
      m.positions[b + 1] - m.positions[a + 1],
      m.positions[b + 2] - m.positions[a + 2],
    ];
    const ac: Vec3 = [
      m.positions[c] - m.positions[a],
      m.positions[c + 1] - m.positions[a + 1],
      m.positions[c + 2] - m.positions[a + 2],
    ];
    const n = v3.norm(v3.cross(ab, ac));
    out[f * 3] = n[0];
    out[f * 3 + 1] = n[1];
    out[f * 3 + 2] = n[2];
  }
  return out;
}

/** 面积加权顶点法线 */
export function vertexNormals(m: Mesh): Float32Array {
  const V = vertexCount(m);
  const out = new Float32Array(V * 3);
  const fn = faceNormals(m);
  for (let f = 0; f < faceCount(m); f++) {
    const a = m.indices[f * 3];
    const b = m.indices[f * 3 + 1];
    const c = m.indices[f * 3 + 2];
    const pa = getVertex(m, a);
    const pb = getVertex(m, b);
    const pc = getVertex(m, c);
    const area2 = v3.len(v3.cross(v3.sub(pb, pa), v3.sub(pc, pa)));
    for (const v of [a, b, c]) {
      out[v * 3] += fn[f * 3] * area2;
      out[v * 3 + 1] += fn[f * 3 + 1] * area2;
      out[v * 3 + 2] += fn[f * 3 + 2] * area2;
    }
  }
  for (let i = 0; i < V; i++) {
    const n = v3.norm([out[i * 3], out[i * 3 + 1], out[i * 3 + 2]]);
    out[i * 3] = n[0];
    out[i * 3 + 1] = n[1];
    out[i * 3 + 2] = n[2];
  }
  return out;
}

/** 每个顶点的混合 Voronoi 面积（Heat method 质量矩阵用） */
export function vertexAreas(m: Mesh): Float32Array {
  const V = vertexCount(m);
  const areas = new Float32Array(V);
  for (let f = 0; f < faceCount(m); f++) {
    const tri = [m.indices[f * 3], m.indices[f * 3 + 1], m.indices[f * 3 + 2]];
    const p = tri.map((v) => getVertex(m, v));
    const e01 = v3.sub(p[1], p[0]);
    const e12 = v3.sub(p[2], p[1]);
    const e20 = v3.sub(p[0], p[2]);
    const l01 = v3.len(e01), l12 = v3.len(e12), l20 = v3.len(e20);
    const area = 0.5 * v3.len(v3.cross(e01, v3.sub(p[2], p[0])));
    // 钝角三角形退化为 1/4、1/4、1/2 拆分，否则按边中点+外心
    const dot0 = v3.dot(v3.scale(e20, -1), e01);
    const dot1 = v3.dot(v3.scale(e01, -1), e12);
    const dot2 = v3.dot(v3.scale(e12, -1), v3.scale(e20, -1));
    let w: Vec3;
    if (dot0 <= 0 || dot1 <= 0 || dot2 <= 0) {
      const obtuse = dot0 <= 0 ? 0 : dot1 <= 1 ? 1 : 2;
      w = obtuse === 0 ? [0.5, 0.25, 0.25] : obtuse === 1 ? [0.25, 0.5, 0.25] : [0.25, 0.25, 0.5];
    } else {
      const w0 = (l01 * l01 * dot2 + l20 * l20 * dot1) / (8 * area * area);
      const w1 = (l12 * l12 * dot0 + l01 * l01 * dot2) / (8 * area * area);
      w = [w0, w1, 1 - w0 - w1];
    }
    for (let k = 0; k < 3; k++) areas[tri[k]] += area * w[k];
  }
  return areas;
}
