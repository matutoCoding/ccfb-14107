import { Mesh, Adjacency, vertexCount, getVertex, setVertex, vertexNormals } from './mesh';
import { geodesicDistance } from './geodesic';
import { v3, smoothstep } from './math';

export type BrushKind = 'pad' | 'release';

export interface BrushStroke {
  id: string;
  kind: BrushKind;
  /** 笔刷中心吸附的顶点 */
  centerVertex: number;
  /** 影响半径（毫米，测地距离） */
  radiusMm: number;
  /** 推压深度（毫米，正数；pad 沿 -N，release 沿 +N） */
  depthMm: number;
  /** pad 自动生成的对侧扩张空腔（存下来便于评估/渲染） */
  contralateral?: {
    centerVertex: number;
    radiusMm: number;
    depthMm: number;
  };
  label: string;
  createdAt: number;
}

export interface SculptResult {
  stroke: BrushStroke;
  /** 每个顶点的法向位移量（毫米，带符号：压为负、释放为正） */
  displacement: Float32Array;
  geodesic: Float32Array;
  elapsedMs: number;
}

/** 测地高斯衰减：d=0 处为 1，d=radius 处约 0.01（3σ 截断），全程沿体表而非直线 */
export function geodesicFalloff(d: Float32Array, radiusMm: number): Float32Array {
  const sigma = radiusMm / 3;
  const out = new Float32Array(d.length);
  for (let i = 0; i < d.length; i++) {
    if (d[i] <= radiusMm) out[i] = Math.exp(-(d[i] * d[i]) / (2 * sigma * sigma));
  }
  return out;
}

/**
 * 应用一笔修型。
 * pad：沿内法线推压 depthMm；release：沿外法线释放。
 * 压垫自动在对侧（镜像 X，患者左右翻转）生成扩张空腔，深度为压垫的 60%、半径 1.15 倍。
 * 位移可叠加，每笔返回带符号位移场，供色谱/撤销使用。
 */
export function applyStroke(
  mesh: Mesh,
  adj: Adjacency,
  input: Omit<BrushStroke, 'id' | 'createdAt' | 'label' | 'contralateral'>,
  options: { autoContralateral?: boolean; useDijkstraFallback?: boolean } = {},
): SculptResult {
  const t0 = performance.now();
  const V = vertexCount(mesh);
  const normals = vertexNormals(mesh);
  const method = options.useDijkstraFallback === false ? 'heat' : 'dijkstra';
  const g0 = geodesicDistance(mesh, input.centerVertex, { method }).distance;
  const w0 = geodesicFalloff(g0, input.radiusMm);

  const displacement = new Float32Array(V);
  const sign = input.kind === 'pad' ? -1 : 1;
  for (let i = 0; i < V; i++) {
    displacement[i] = sign * input.depthMm * w0[i];
  }

  let contra: BrushStroke['contralateral'];
  // 对侧自动扩张空腔：以 X=0 冠状面镜像找最近体表顶点
  if (options.autoContralateral !== false && input.kind === 'pad') {
    const center = getVertex(mesh, input.centerVertex);
    const mirrored = [-center[0], center[1], center[2]] as const;
    const target = nearestVertexByMirror(mesh, mirrored);
    const g1 = geodesicDistance(mesh, target, { method }).distance;
    const cRadius = input.radiusMm * 1.15;
    const cDepth = input.depthMm * 0.6;
    const w1 = geodesicFalloff(g1, cRadius);
    for (let i = 0; i < V; i++) {
      // 空腔沿 +N 向外，与压垫重叠处以压垫优先
      displacement[i] = displacement[i] !== 0 ? displacement[i] : cDepth * w1[i];
    }
    contra = { centerVertex: target, radiusMm: cRadius, depthMm: cDepth };
  }

  for (let i = 0; i < V; i++) {
    if (displacement[i] === 0) continue;
    const p = getVertex(mesh, i);
    const n: [number, number, number] = [normals[i * 3], normals[i * 3 + 1], normals[i * 3 + 2]];
    setVertex(mesh, i, v3.add(p, v3.scale(n, displacement[i])));
  }

  const stroke: BrushStroke = {
    ...input,
    id: `stroke_${Math.round(performance.now() * 1000)}_${input.centerVertex}`,
    label: input.kind === 'pad' ? `压垫@${input.centerVertex}` : `释放区@${input.centerVertex}`,
    createdAt: Date.now(),
    contralateral: contra,
  };
  return { stroke, displacement, geodesic: g0, elapsedMs: performance.now() - t0 };
}

/** 镜像点最近顶点（先按 X 符号筛对侧，再取欧氏最近，规模大可换 BVH） */
function nearestVertexByMirror(mesh: Mesh, target: readonly number[]): number {
  const V = vertexCount(mesh);
  let best = -1;
  let bestD = Infinity;
  for (let v = 0; v < V; v++) {
    const p = getVertex(mesh, v);
    const d = (p[0] - target[0]) ** 2 + (p[1] - target[1]) ** 2 + (p[2] - target[2]) ** 2;
    if (d < bestD) { bestD = d; best = v; }
  }
  return best;
}

/** 撤销一笔：用反向位移恢复（法线按当前网格重算，误差在笔刷平滑尺度内可忽略） */
export function revertStroke(mesh: Mesh, displacement: Float32Array): void {
  const normals = vertexNormals(mesh);
  for (let i = 0; i < vertexCount(mesh); i++) {
    if (displacement[i] === 0) continue;
    const p = getVertex(mesh, i);
    const n: [number, number, number] = [normals[i * 3], normals[i * 3 + 1], normals[i * 3 + 2]];
    setVertex(mesh, i, v3.sub(p, v3.scale(n, displacement[i])));
  }
}

/** 修型前后差值（毫米，沿当前法线方向的有符号位移，正值=向外释放，负值=向内推压） */
export function signedDifference(before: Mesh, after: Mesh): Float32Array {
  const V = vertexCount(after);
  const normals = vertexNormals(after);
  const out = new Float32Array(V);
  for (let i = 0; i < V; i++) {
    const a = getVertex(before, i);
    const b = getVertex(after, i);
    const n: [number, number, number] = [normals[i * 3], normals[i * 3 + 1], normals[i * 3 + 2]];
    const d = v3.sub(b, a);
    out[i] = v3.dot(d, n);
  }
  return out;
}

/** 差值转双色色谱（负=蓝→压，正=红→释放，零=白），输出顶点 RGB */
export function differenceColormap(diff: Float32Array, maxMm = 10): Float32Array {
  const colors = new Float32Array(diff.length * 3);
  for (let i = 0; i < diff.length; i++) {
    const t = Math.min(1, Math.abs(diff[i]) / maxMm);
    if (diff[i] < -1e-4) {
      // 白 → 蓝
      colors[i * 3] = 1 - t;
      colors[i * 3 + 1] = 1 - t * 0.6;
      colors[i * 3 + 2] = 1;
    } else if (diff[i] > 1e-4) {
      // 白 → 红
      colors[i * 3] = 1;
      colors[i * 3 + 1] = 1 - t * 0.7;
      colors[i * 3 + 2] = 1 - t;
    } else {
      colors[i * 3] = colors[i * 3 + 1] = colors[i * 3 + 2] = 0.92;
    }
  }
  return colors;
}

/** 沿高度（Y）取截面轮廓：返回该高度 ±band 内的环带顶点，按角度排序成闭合折线 */
export function crossSection(mesh: Mesh, heightMm: number, bandMm = 3): { points: [number, number][]; ring: number[] } {
  const V = vertexCount(mesh);
  const ring: number[] = [];
  for (let v = 0; v < V; v++) {
    const p = getVertex(mesh, v);
    if (Math.abs(p[1] - heightMm) <= bandMm) ring.push(v);
  }
  ring.sort((a, b) => {
    const pa = getVertex(mesh, a);
    const pb = getVertex(mesh, b);
    return Math.atan2(pa[2], pa[0]) - Math.atan2(pb[2], pb[0]);
  });
  const points = ring.map((v) => {
    const p = getVertex(mesh, v);
    return [p[0], p[2]] as [number, number];
  });
  return { points, ring };
}

/** 压垫边缘的平滑过渡带工具（UI 显示笔刷范围时复用） */
export function brushRingWeight(d: number, radius: number): number {
  return smoothstep(radius * 0.9, radius, d);
}
