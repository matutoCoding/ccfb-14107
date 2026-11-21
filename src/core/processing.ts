// 扫描预处理：杂点/扶架去除、自动补洞、解剖标记点坐标系摆正
import { TriangleMesh } from './mesh';
import { Vec3, add, cross, frame4, invert4, normalize, scale, sub, transformPoint4 } from './math';

export interface CleaningStats {
  removedComponents: number;
  removedFaces: number;
  filledHoles: number;
  filledTriangles: number;
}

/**
 * 连通域分析 + 杂点剔除：
 * 保留面数占比最大的主体（躯干），手臂/扶架等离散小连通域整体移除
 */
export function removeSmallComponents(mesh: TriangleMesh, minFaceRatio = 0.02): { mesh: TriangleMesh; stats: CleaningStats } {
  const faceCount = mesh.faceCount;
  const adj = mesh.faceAdjacency();
  const seen = new Uint8Array(faceCount);
  const components: number[][] = [];
  for (let f = 0; f < faceCount; f++) {
    if (seen[f]) continue;
    const stack = [f];
    const comp: number[] = [];
    seen[f] = 1;
    while (stack.length) {
      const cur = stack.pop()!;
      comp.push(cur);
      for (let k = 0; k < 3; k++) {
        const nb = adj[cur * 3 + k];
        if (nb >= 0 && !seen[nb]) {
          seen[nb] = 1;
          stack.push(nb);
        }
      }
    }
    components.push(comp);
  }
  components.sort((a, b) => b.length - a.length);
  const threshold = Math.max(20, faceCount * minFaceRatio);
  const keep = new Set<number>();
  let removedFaces = 0;
  let removedComponents = 0;
  for (const comp of components) {
    if (comp.length >= threshold || comp === components[0]) comp.forEach((f) => keep.add(f));
    else {
      removedFaces += comp.length;
      removedComponents++;
    }
  }
  // 重打包顶点/索引
  const oldToNew = new Map<number, number>();
  const pos: number[] = [];
  const idx: number[] = [];
  for (let f = 0; f < faceCount; f++) {
    if (!keep.has(f)) continue;
    const o = f * 3;
    const tri = [mesh.indices[o], mesh.indices[o + 1], mesh.indices[o + 2]];
    const mapped = tri.map((v) => {
      let nv = oldToNew.get(v);
      if (nv === undefined) {
        nv = pos.length / 3;
        oldToNew.set(v, nv);
        pos.push(mesh.positions[v * 3], mesh.positions[v * 3 + 1], mesh.positions[v * 3 + 2]);
      }
      return nv;
    });
    idx.push(...mapped);
  }
  return {
    mesh: new TriangleMesh(Float32Array.from(pos), Uint32Array.from(idx)),
    stats: { removedComponents, removedFaces, filledHoles: 0, filledTriangles: 0 },
  };
}

/**
 * 基于边界环追踪 + 扇形三角化的自动补洞
 * 适用于扫描破洞（边界环顶点 < 200 的单连通孔）；大洞在 UI 提示手工处理
 */
export function fillHoles(mesh: TriangleMesh, maxLoopSize = 300): { mesh: TriangleMesh; filledHoles: number; filledTriangles: number } {
  const adj = mesh.faceAdjacency();
  // 无向边界边集合（只出现一次的边）
  const n = mesh.vertexCount;
  const keyOf = (a: number, b: number) => Math.min(a, b) * n + Math.max(a, b);
  const edgeCount = new Map<number, [number, number]>();
  for (let f = 0; f < mesh.faceCount; f++) {
    const [a, b, c] = mesh.face(f);
    const edges: Array<[number, number]> = [
      [b, c],
      [c, a],
      [a, b],
    ];
    for (let k = 0; k < 3; k++) {
      if (adj[f * 3 + k] !== -1) continue;
      const [u, v] = edges[k];
      edgeCount.set(keyOf(u, v), [u, v]);
    }
  }
  // 有向边界边：next[u] = v 表示边界边 u→v（沿面绕序方向）
  // faceAdjacency 的 slot 约定为对边：slot0=[b,c]、slot1=[c,a]、slot2=[a,b]
  // 流形环上每个边界顶点恰有一条出边和一条入边
  const next = new Map<number, number>();
  for (let f = 0; f < mesh.faceCount; f++) {
    const [a, b, c] = mesh.face(f);
    const edges: Array<[number, number]> = [
      [b, c],
      [c, a],
      [a, b],
    ];
    for (let k = 0; k < 3; k++) {
      if (adj[f * 3 + k] !== -1) continue;
      const [u, v] = edges[k];
      if (!next.has(u)) next.set(u, v);
    }
  }
  // 逐环追踪有向边链
  const usedVertices = new Set<number>();
  const loops: number[][] = [];
  for (const start of next.keys()) {
    if (usedVertices.has(start)) continue;
    const loop: number[] = [];
    let cur = start;
    for (;;) {
      if (usedVertices.has(cur)) break;
      usedVertices.add(cur);
      loop.push(cur);
      const nv = next.get(cur);
      if (nv === undefined || nv === start) break;
      cur = nv;
      if (loop.length > maxLoopSize) break;
    }
    const closed = next.get(cur) === start;
    if (closed && loop.length >= 3 && loop.length <= maxLoopSize) {
      loops.push(loop);
    }
  }

  const positions = Array.from(mesh.positions);
  const indices = Array.from(mesh.indices);
  let filledTriangles = 0;
  let filledHoles = 0;
  for (const loop of loops) {
    const center: Vec3 = [0, 0, 0];
    for (const v of loop) {
      center[0] += positions[v * 3];
      center[1] += positions[v * 3 + 1];
      center[2] += positions[v * 3 + 2];
    }
    center[0] /= loop.length;
    center[1] /= loop.length;
    center[2] /= loop.length;
    const cv = positions.length / 3;
    positions.push(center[0], center[1], center[2]);
    for (let k = 0; k < loop.length; k++) {
      const a = loop[k];
      const b = loop[(k + 1) % loop.length];
      indices.push(a, b, cv);
      filledTriangles++;
    }
    filledHoles++;
  }
  return {
    mesh: new TriangleMesh(Float32Array.from(positions), Uint32Array.from(indices)),
    filledHoles,
    filledTriangles,
  };
}

function sub3(a: Vec3, b: Vec3): Vec3 {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}
function norm3(a: Vec3): Vec3 {
  const l = Math.hypot(a[0], a[1], a[2]) || 1;
  return [a[0] / l, a[1] / l, a[2] / l];
}

/** 解剖标记点（髂前上棘左右、腋窝左右、颈切迹等） */
export interface AnatomicalLandmarks {
  asiLeft: Vec3;   // 左髂前上棘
  asiRight: Vec3;  // 右髂前上棘
  axillaLeft: Vec3;
  axillaRight: Vec3;
  /** 最近顶点索引（便于拖改后同步） */  vertexHints?: Record<string, number>;
}

/**
 * 由标记点构建解剖坐标系并摆正：
 * Y 轴 = 腋窝中点 -> ASIS 中点（躯干长轴）
 * X 轴 = 右 ASIS -> 左 ASIS（骨盆横轴）
 * Z 轴 = X × Y（前后方向）
 */
export function buildAlignmentFrame(lm: AnatomicalLandmarks): { frame: Float32Array; inverse: Float32Array; origin: Vec3 } {
  const pelvisMid = scale(add(lm.asiLeft, lm.asiRight), 0.5);
  const axillaMid = scale(add(lm.axillaLeft, lm.axillaRight), 0.5);
  const yAxis = normalize(sub(pelvisMid, axillaMid)); // 指向脚
  const xAxisRaw = sub(lm.asiLeft, lm.asiRight);
  // 正交化：x' = x - (x·y)y
  const xAxis = normalize(sub(xAxisRaw, scale(yAxis, dot3(xAxisRaw, yAxis))));
  const zAxis = normalize(cross(xAxis, yAxis));
  const origin = pelvisMid;
  const frame = frame4(xAxis, yAxis, zAxis, origin);
  return { frame, inverse: invert4(frame), origin };
}

function dot3(a: Vec3, b: Vec3): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

export function alignMesh(mesh: TriangleMesh, lm: AnatomicalLandmarks): TriangleMesh {
  const { inverse } = buildAlignmentFrame(lm);
  return mesh.transformed(inverse);
}

/**
 * 自动猜测标记点位置（首版用启发式，用户可拖动修正）：
 * ASIS：网格下 1/3 高度处，左右最宽点；腋窝：上 1/3 处宽度收窄的拐点
 */
export function guessLandmarks(mesh: TriangleMesh): AnatomicalLandmarks {
  const { min, max } = mesh.bbox();
  const height = max[1] - min[1];
  const band = (y0: number, y1: number) => {
    const pts: Array<{ i: number; p: Vec3 }> = [];
    for (let i = 0; i < mesh.vertexCount; i++) {
      const y = mesh.positions[i * 3 + 1];
      if (y >= min[1] + height * y0 && y <= min[1] + height * y1) {
        pts.push({ i, p: mesh.vertex(i) });
      }
    }
    return pts;
  };
  const extremes = (pts: Array<{ i: number; p: Vec3 }>) => {
    let left = pts[0],
      right = pts[0];
    for (const item of pts) {
      if (item.p[0] > left.p[0]) left = item;
      if (item.p[0] < right.p[0]) right = item;
    }
    return { left, right };
  };
  const asis = extremes(band(0.12, 0.28));
  const axilla = extremes(band(0.62, 0.78));
  return {
    asiLeft: asis.left.p,
    asiRight: asis.right.p,
    axillaLeft: axilla.left.p,
    axillaRight: axilla.right.p,
    vertexHints: {
      asiLeft: asis.left.i,
      asiRight: asis.right.i,
      axillaLeft: axilla.left.i,
      axillaRight: axilla.right.i,
    },
  };
}

/** 将一个世界坐标点吸附到网格最近顶点（标记点拖拽时用） */
export function snapToVertex(mesh: TriangleMesh, point: Vec3): { vertex: number; position: Vec3 } {
  let best = 0;
  let bestD = Infinity;
  for (let i = 0; i < mesh.vertexCount; i++) {
    const dx = mesh.positions[i * 3] - point[0];
    const dy = mesh.positions[i * 3 + 1] - point[1];
    const dz = mesh.positions[i * 3 + 2] - point[2];
    const d = dx * dx + dy * dy + dz * dz;
    if (d < bestD) {
      bestD = d;
      best = i;
    }
  }
  return { vertex: best, position: mesh.vertex(best) };
}

export { transformPoint4 };
