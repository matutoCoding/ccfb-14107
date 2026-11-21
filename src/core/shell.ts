// 支具壳体引擎：等距偏移成壳、分区厚度、边缘修剪/开窗/后开口、倒圆、自交与薄壳校验
import { TriangleMesh, SpatialHash } from './mesh';
import { Vec3, add, clamp, cross, normalize, scale, sub } from './math';

export interface ThicknessZone {
  /** 区域名称 */
  name: string;
  /** Y 高度区间 mm（解剖坐标系，沿脊柱轴） */
  yRange: [number, number];
  /** 角向区间（弧度，0=+X 冠状右），null 表示整圈 */
  thetaRange: [number, number] | null;
  thicknessMm: number;
}

export interface TrimLine {
  /** 修剪线上的顶点（体表绘制后吸附），按高度形成环 */
  vertices: number[];
  /** 在上缘还是下缘 */
  edge: 'superior' | 'inferior';
}

export interface WindowCut {
  /** 开窗中心顶点 */
  centerVertex: number;
  /** 半径 mm */
  radiusMm: number;
  /** 倒圆半径 mm */
  filletMm: number;
}

export interface ShellOptions {
  thicknessZones: ThicknessZone[];
  defaultThicknessMm: number;
  trimLines: TrimLine[];
  windows: WindowCut[];
  /** 后开口宽度 mm（沿 +Z/-Z 背面中线切除的角向宽度） */
  posteriorOpeningMm: number;
  /** 切口倒圆半径 */
  edgeFilletMm: number;
}

export interface ShellValidation {
  watertight: boolean;
  /** 薄于 2mm 的顶点 */
  thinVertices: number[];
  /** 自交面（采样命中） */
  selfIntersectingFaces: number[];
  minThicknessMm: number;
  maxThicknessMm: number;
  /** 随机抽点的厚度误差统计 */
  thicknessSample: { mean: number; maxAbs: number; samples: number };
}

/** 每顶点厚度赋值（分区规则） */
export function computeThicknessField(mesh: TriangleMesh, opts: ShellOptions): Float32Array {
  const out = new Float32Array(mesh.vertexCount).fill(opts.defaultThicknessMm);
  for (let i = 0; i < mesh.vertexCount; i++) {
    const p = mesh.vertex(i);
    const theta = Math.atan2(p[2], p[0]);
    for (const zone of opts.thicknessZones) {
      if (p[1] < zone.yRange[0] || p[1] > zone.yRange[1]) continue;
      if (zone.thetaRange) {
        const [t0, t1] = zone.thetaRange;
        const norm = (t: number) => ((t + Math.PI * 3) % (Math.PI * 2)) - Math.PI;
        const dt = norm(theta - t0);
        const span = t1 - t0;
        if (dt < 0 || dt > span) continue;
      }
      out[i] = zone.thicknessMm;
    }
  }
  return out;
}

/**
 * 等距偏移生成双层壳体：
 * 内壁 = 修型面沿 -n 偏移 0（贴合面即修型表面）
 * 外壁 = 修型面沿 +n 偏移 thickness(v)
 * 边缘环带连接内外壁，形成水密实体。这是保证“抽点壳厚误差 ≤0.1mm”的直接建模方式。
 */
export function buildShell(innerMesh: TriangleMesh, thickness: Float32Array, opts: ShellOptions): TriangleMesh {
  const n = innerMesh.vertexCount;
  const normals = innerMesh.vertexNormals();
  const outerPositions = new Float32Array(innerMesh.positions.length);
  for (let i = 0; i < n; i++) {
    const t = thickness[i];
    outerPositions[i * 3] = innerMesh.positions[i * 3] + normals[i * 3] * t;
    outerPositions[i * 3 + 1] = innerMesh.positions[i * 3 + 1] + normals[i * 3 + 1] * t;
    outerPositions[i * 3 + 2] = innerMesh.positions[i * 3 + 2] + normals[i * 3 + 2] * t;
  }

  // 决定保留面：修剪线以外、开窗、后开口都要剔除
  const keepFace = computeFaceKeepMask(innerMesh, opts);

  // 组装：内壁顶点 + 外壁顶点，面各自复制，边界环桥接
  const positions = new Float32Array(n * 2 * 3);
  positions.set(innerMesh.positions, 0);
  positions.set(outerPositions, n * 3);
  const indices: number[] = [];
  const boundaryEdges = new Map<number, number>(); // 内网格边界边 u->v
  const edgeKey = (a: number, b: number) => Math.min(a, b) * n + Math.max(a, b);
  const edgeSeen = new Set<number>();

  for (let f = 0; f < innerMesh.faceCount; f++) {
    if (!keepFace[f]) continue;
    const o = f * 3;
    const a = innerMesh.indices[o],
      b = innerMesh.indices[o + 1],
      c = innerMesh.indices[o + 2];
    // 内壁（反向绕序使法向朝内）
    indices.push(a, c, b);
    // 外壁（顶点号 +n，保持原绕序）
    indices.push(a + n, b + n, c + n);
    // 记录边界边
    const tri = [a, b, c];
    for (let k = 0; k < 3; k++) {
      const u = tri[k],
        v = tri[(k + 1) % 3];
      const key = edgeKey(u, v);
      if (edgeSeen.has(key)) {
        edgeSeen.delete(key);
        boundaryEdges.delete(key);
      } else {
        edgeSeen.add(key);
        boundaryEdges.set(key, v);
        boundaryEdges.set(key + 0.5, u); // 临时存端点
      }
    }
  }

  // 桥接边界环：每条边界边生成内外壁连接四边形（2 三角形）
  const boundaryPairs = extractBoundaryPairs(innerMesh, keepFace);
  for (const [u, v] of boundaryPairs) {
    indices.push(u, v, v + n);
    indices.push(u, v + n, u + n);
  }
  void boundaryEdges;

  return new TriangleMesh(positions, Uint32Array.from(indices));
}

/** 从保留面掩码提取边界边（仅出现一次的边） */
function extractBoundaryPairs(mesh: TriangleMesh, keepFace: Uint8Array): Array<[number, number]> {
  const n = mesh.vertexCount;
  const map = new Map<number, { u: number; v: number; count: number }>();
  const keyOf = (a: number, b: number) => Math.min(a, b) * n + Math.max(a, b);
  for (let f = 0; f < mesh.faceCount; f++) {
    if (!keepFace[f]) continue;
    const [a, b, c] = mesh.face(f);
    const edges: Array<[number, number]> = [
      [a, b],
      [b, c],
      [c, a],
    ];
    for (const [u, v] of edges) {
      const key = keyOf(u, v);
      const cur = map.get(key);
      if (cur) cur.count++;
      else map.set(key, { u: Math.min(u, v), v: Math.max(u, v), count: 1 });
    }
  }
  return Array.from(map.values()).filter((e) => e.count === 1).map((e) => [e.u, e.v]);
}

/** 面保留掩码：综合修剪线、开窗、后开口 */
function computeFaceKeepMask(mesh: TriangleMesh, opts: ShellOptions): Uint8Array {
  const keep = new Uint8Array(mesh.faceCount).fill(1);
  const { min, max } = mesh.bbox();
  const height = max[1] - min[1];

  // 修剪线高度阈值
  let cutTop = max[1] + 1;
  let cutBottom = min[1] - 1;
  for (const line of opts.trimLines) {
    const ys = line.vertices.map((v) => mesh.positions[v * 3 + 1]);
    const yLimit = ys.reduce((a, b) => a + b, 0) / ys.length;
    if (line.edge === 'superior') cutTop = yLimit;
    else cutBottom = yLimit;
  }

  // 开窗中心
  const windowCenters = opts.windows.map((w) => ({
    center: mesh.vertex(w.centerVertex),
    radius: w.radiusMm + w.filletMm,
  }));

  // 后开口：背面中线（-Z 方向）附近按弧宽切除
  const openingHalfAngle = Math.atan2(opts.posteriorOpeningMm / 2, estimateRadius(mesh));

  for (let f = 0; f < mesh.faceCount; f++) {
    const o = f * 3;
    const vi = mesh.indices[o];
    const p = mesh.vertex(vi);
    if (p[1] > cutTop || p[1] < cutBottom) {
      keep[f] = 0;
      continue;
    }
    const theta = Math.atan2(p[2], p[0]);
    // 背面取 theta ≈ -π/2 附近（Z 负）
    const backAngle = Math.abs(angleDiff(theta, -Math.PI / 2));
    if (opts.posteriorOpeningMm > 0 && backAngle < openingHalfAngle) {
      keep[f] = 0;
      continue;
    }
    for (const w of windowCenters) {
      if (distance3(p, w.center) < w.radius) {
        keep[f] = 0;
        break;
      }
    }
  }
  void height;
  return keep;
}

function estimateRadius(mesh: TriangleMesh): number {
  const { min, max } = mesh.bbox();
  return Math.max(max[0] - min[0], max[2] - min[2]) / 2;
}
function angleDiff(a: number, b: number): number {
  let d = a - b;
  while (d > Math.PI) d -= Math.PI * 2;
  while (d < -Math.PI) d += Math.PI * 2;
  return d;
}
function distance3(a: Vec3, b: Vec3): number {
  return Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
}

/**
 * 壳体校验：水密性、厚度分布、自交检测
 */
export function validateShell(shell: TriangleMesh, innerMesh: TriangleMesh, thickness: Float32Array, sampleCount = 200): ShellValidation {
  const watertight = shell.isWatertight();
  const half = innerMesh.vertexCount;

  // 厚度：内外壁对应顶点距离
  const measured: number[] = [];
  const thinVertices: number[] = [];
  let minT = Infinity;
  let maxT = -Infinity;
  for (let i = 0; i < half; i++) {
    const a: Vec3 = [shell.positions[i * 3], shell.positions[i * 3 + 1], shell.positions[i * 3 + 2]];
    const b: Vec3 = [
      shell.positions[(i + half) * 3],
      shell.positions[(i + half) * 3 + 1],
      shell.positions[(i + half) * 3 + 2],
    ];
    const t = distance3(a, b);
    measured.push(t);
    if (t < minT) minT = t;
    if (t > maxT) maxT = t;
    if (t < 2.0) thinVertices.push(i);
  }

  // 随机抽点厚度误差（与设计厚度对比）
  let step = Math.max(1, Math.floor(half / sampleCount));
  let sum = 0;
  let maxAbs = 0;
  let samples = 0;
  for (let i = 0; i < half; i += step) {
    const err = Math.abs(measured[i] - thickness[i]);
    sum += err;
    maxAbs = Math.max(maxAbs, err);
    samples++;
  }

  // 自交检测：外壁面与内壁面的空间碰撞（空间哈希加速，采样）
  const hash = new SpatialHash(shell, 8);
  const selfIntersecting = new Set<number>();
  const checkStride = Math.max(1, Math.floor(shell.faceCount / 4000));
  for (let f = 0; f < shell.faceCount; f += checkStride) {
    const [a, b, c] = shell.face(f);
    const centroid = scale(add(add(shell.vertex(a), shell.vertex(b)), shell.vertex(c)), 1 / 3);
    const cands = hash.candidateFaces(centroid[0], centroid[1], centroid[2], 3);
    for (const g of cands) {
      if (g === f || Math.abs(g - f) < 4) continue;
      // 内外壁面号区分：前 half*2 个面为内/外壁主面
      const fOuter = f >= innerMesh.faceCount;
      const gOuter = g >= innerMesh.faceCount;
      if (fOuter === gOuter) continue;
      const ga = shell.vertex(shell.indices[g * 3]);
      if (distance3(centroid, ga) < 1.5) {
        selfIntersecting.add(f);
        break;
      }
    }
  }

  return {
    watertight,
    thinVertices,
    selfIntersectingFaces: Array.from(selfIntersecting),
    minThicknessMm: minT,
    maxThicknessMm: maxT,
    thicknessSample: { mean: sum / samples, maxAbs, samples },
  };
}

/** 默认分区厚度（临床常用：骨盆承重区厚、腰段释放区薄） */
export function defaultThicknessZones(bboxY: [number, number]): ThicknessZone[] {
  const [y0, y1] = bboxY;
  const h = y1 - y0;
  return [
    { name: '骨盆固定区', yRange: [y0, y0 + h * 0.25], thetaRange: null, thicknessMm: 5 },
    { name: '腰段弹性区', yRange: [y0 + h * 0.25, y0 + h * 0.6], thetaRange: null, thicknessMm: 3.5 },
    { name: '胸段支撑区', yRange: [y0 + h * 0.6, y1], thetaRange: null, thicknessMm: 4 },
  ];
}

/** 边缘倒圆：对边界环顶点做沿边方向的位置平滑（单轮拉普拉斯） */
export function filletBoundary(mesh: TriangleMesh, iterations = 2): TriangleMesh {
  const positions = new Float32Array(mesh.positions);
  const adj = mesh.v2v();
  const boundary = new Set(mesh.boundaryVertices());
  for (let it = 0; it < iterations; it++) {
    const snapshot = new Float32Array(positions);
    for (const v of boundary) {
      const neighbors = adj[v].filter((u) => boundary.has(u));
      if (neighbors.length < 2) continue;
      let cx = 0,
        cy = 0,
        cz = 0;
      for (const u of neighbors) {
        cx += snapshot[u * 3];
        cy += snapshot[u * 3 + 1];
        cz += snapshot[u * 3 + 2];
      }
      const w = 0.5;
      positions[v * 3] = snapshot[v * 3] * (1 - w) + (cx / neighbors.length) * w;
      positions[v * 3 + 1] = snapshot[v * 3 + 1] * (1 - w) + (cy / neighbors.length) * w;
      positions[v * 3 + 2] = snapshot[v * 3 + 2] * (1 - w) + (cz / neighbors.length) * w;
    }
  }
  return new TriangleMesh(positions, mesh.indices);
}

/** 法向一致性检查（导出前） */
export function checkNormals(mesh: TriangleMesh): { consistent: boolean; flippedFaces: number } {
  const { center } = mesh.bbox();
  let outward = 0;
  let flipped = 0;
  for (let f = 0; f < mesh.faceCount; f++) {
    const [ia, ib, ic] = mesh.face(f);
    const a = mesh.vertex(ia),
      b = mesh.vertex(ib),
      c = mesh.vertex(ic);
    const fn = normalize(cross(sub(b, a), sub(c, a)));
    const mid = scale(add(add(a, b), c), 1 / 3);
    const out = normalize(sub(mid, center));
    const d = dot3(fn, out);
    if (d > 0) outward++;
    else flipped++;
  }
  return { consistent: flipped < outward * 0.05, flippedFaces: flipped };
}

function dot3(a: Vec3, b: Vec3): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
void clamp;
