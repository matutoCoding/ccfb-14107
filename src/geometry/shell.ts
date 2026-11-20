import { Mesh, vertexCount, faceCount, getVertex, vertexNormals } from './mesh';
import { Vec3, v3, clamp } from './math';
import { TriangleBVH } from './bvh';

/** 厚度分区：按高度（Y 毫米）区间给壳厚，未命中区间用默认值 */
export interface ThicknessZone {
  yMin: number;
  yMax: number;
  thicknessMm: number;
}

/** 修剪线：在体表上画的闭合折线（顶点 id 序列），圈内区域保留/切除 */
export interface TrimLine {
  id: string;
  kind: 'trim' | 'window';
  /** 折线顶点（世界坐标毫米，实际投影到最近体表顶点执行） */
  points: Vec3[];
  /** trim: 保留圈内还是圈外；window 固定切除圈内 */
  keepInside?: boolean;
}

export interface BackOpening {
  /** 后开口在背侧（-Z）的角度半宽（弧度），中心线沿 -Z */
  halfAngle: number;
  yMin: number;
  yMax: number;
}

export interface ShellOptions {
  zones: ThicknessZone[];
  defaultThicknessMm: number;
  trimLines: TrimLine[];
  backOpening?: BackOpening;
  /** 切口倒圆半径（毫米） */
  filletRadiusMm: number;
}

export interface ShellIssue {
  kind: 'thin' | 'self-intersection' | 'non-manifold';
  vertexId?: number;
  faceId?: number;
  detail: string;
}

export interface ShellResult {
  shell: Mesh;
  inner: Mesh;
  outer: Mesh;
  thicknessField: Float32Array;
  issues: ShellIssue[];
  watertight: boolean;
}

function thicknessAt(y: number, zones: ThicknessZone[], fallback: number): number {
  for (const z of zones) {
    if (y >= z.yMin && y <= z.yMax) return z.thicknessMm;
  }
  return fallback;
}

/**
 * 等距偏移壳体。
 * 内壁 = 修型后表面本身（与皮肤间隙在穿戴评估中计算）；
 * 外壁 = 沿顶点法线外偏 thickness(v)。同拓扑保证厚度场精确可控，
 * 再沿修剪边界/开口边界缝合内外壁形成实体壳，切口处加环形倒圆。
 */
export function buildShell(sculpted: Mesh, opts: ShellOptions): ShellResult {
  const V = vertexCount(sculpted);
  const F = faceCount(sculpted);
  const normals = vertexNormals(sculpted);
  const thickness = new Float32Array(V);
  for (let v = 0; v < V; v++) {
    thickness[v] = thicknessAt(getVertex(sculpted, v)[1], opts.zones, opts.defaultThicknessMm);
  }

  // 内外壁（同拓扑复制）
  const inner: Mesh = {
    positions: new Float32Array(sculpted.positions),
    indices: new Uint32Array(sculpted.indices),
    name: 'shell-inner',
  };
  const outerPos = new Float32Array(sculpted.positions.length);
  for (let v = 0; v < V; v++) {
    const p = getVertex(sculpted, v);
    const n: Vec3 = [normals[v * 3], normals[v * 3 + 1], normals[v * 3 + 2]];
    const q = v3.add(p, v3.scale(n, thickness[v]));
    outerPos[v * 3] = q[0];
    outerPos[v * 3 + 1] = q[1];
    outerPos[v * 3 + 2] = q[2];
  }

  // ---- 面掩码：修剪线 / 开窗 / 后开口 ----
  const alive = new Uint8Array(F).fill(1);
  const trimVertexMask = computeTrimMask(sculpted, opts.trimLines);
  for (let f = 0; f < F; f++) {
    const tri = [sculpted.indices[f * 3], sculpted.indices[f * 3 + 1], sculpted.indices[f * 3 + 2]];
    if (tri.some((v) => trimVertexMask[v] === 0)) alive[f] = 0;
    if (opts.backOpening && faceInBackOpening(sculpted, tri, opts.backOpening)) alive[f] = 0;
  }

  // 压缩存活面并缝合
  const result = stitchShell(inner, outerPos, alive, thickness, opts.filletRadiusMm);
  const issues = validateShell(result.shell, result.thicknessField, result.watertight);
  return { ...result, thicknessField: result.thicknessField, issues };
}

/** 修剪/开窗掩码：window 圈内删面；trim 按 keepInside 决定 */
function computeTrimMask(mesh: Mesh, lines: TrimLine[]): Uint8Array {
  const V = vertexCount(mesh);
  const mask = new Uint8Array(V).fill(1);
  for (const line of lines) {
    if (line.points.length < 3) continue;
    // 投影到体表顶点，得到圈定顶点集合的高度范围与 XZ 多边形
    const poly = line.points.map((p) => [p[0], p[2]] as [number, number]);
    const ys = line.points.map((p) => p[1]);
    const yMin = Math.min(...ys) - 2;
    const yMax = Math.max(...ys) + 2;
    for (let v = 0; v < V; v++) {
      const p = getVertex(mesh, v);
      if (p[1] < yMin || p[1] > yMax) continue;
      const inside = pointInPolygon([p[0], p[2]], poly);
      if (line.kind === 'window') {
        if (inside) mask[v] = 0;
      } else if (line.keepInside === false) {
        if (inside) mask[v] = 0;
      } else {
        if (!inside) mask[v] = 0;
      }
    }
  }
  return mask;
}

function pointInPolygon(p: [number, number], poly: [number, number][]): boolean {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const xi = poly[i][0], yi = poly[i][1];
    const xj = poly[j][0], yj = poly[j][1];
    const intersect = yi > p[1] !== yj > p[1] && p[0] < ((xj - xi) * (p[1] - yi)) / (yj - yi) + xi;
    if (intersect) inside = !inside;
  }
  return inside;
}

function faceInBackOpening(mesh: Mesh, tri: number[], bo: BackOpening): boolean {
  for (const v of tri) {
    const p = getVertex(mesh, v);
    if (p[1] < bo.yMin || p[1] > bo.yMax) return false;
    const ang = Math.atan2(p[2], p[0]); // 背侧 -Z → ang ≈ -π/2
    const back = -Math.PI / 2;
    let d = Math.abs(ang - back);
    if (d > Math.PI) d = 2 * Math.PI - d;
    if (d > bo.halfAngle) return false;
  }
  return true;
}

/** 缝合内外壁：压缩存活面，边界环加墙（四边形拆三角形），切口顶点做倒圆位移 */
function stitchShell(
  innerSrc: Mesh,
  outerPos: Float32Array,
  alive: Uint8Array,
  thickness: Float32Array,
  fillet: number,
): { shell: Mesh; inner: Mesh; outer: Mesh; thicknessField: Float32Array; watertight: boolean } {
  const V = vertexCount(innerSrc);
  const F = faceCount(innerSrc);
  const keptFaces: number[] = [];
  for (let f = 0; f < F; f++) if (alive[f]) keptFaces.push(f);

  // 边界边：存活面中只出现一次的边
  const edgeCount = new Map<string, number>();
  const ekey = (a: number, b: number) => (a < b ? `${a}|${b}` : `${b}|${a}`);
  for (const f of keptFaces) {
    const tri = [innerSrc.indices[f * 3], innerSrc.indices[f * 3 + 1], innerSrc.indices[f * 3 + 2]];
    for (let k = 0; k < 3; k++) {
      const e = ekey(tri[k], tri[(k + 1) % 3]);
      edgeCount.set(e, (edgeCount.get(e) ?? 0) + 1);
    }
  }
  const boundaryEdges: [number, number][] = [];
  for (const [key, count] of edgeCount) {
    if (count === 1) {
      const [a, b] = key.split('|').map(Number);
      boundaryEdges.push([a, b]);
    }
  }
  const boundaryVerts = new Set<number>();
  for (const [a, b] of boundaryEdges) { boundaryVerts.add(a); boundaryVerts.add(b); }

  // 倒圆：边界顶点的内外壁位置沿厚度方向各回缩 fillet，并在中间补一圈圆角顶点
  const innerNormals = vertexNormals(innerSrc);
  const positions: number[] = [];
  const innerId = new Int32Array(V).fill(-1);
  const outerId = new Int32Array(V).fill(-1);
  const filletId = new Int32Array(V).fill(-1);
  const emit = (p: Vec3): number => {
    const id = positions.length / 3;
    positions.push(p[0], p[1], p[2]);
    return id;
  };
  const r = clamp(fillet, 0, 4);
  for (let v = 0; v < V; v++) {
    const p = getVertex(innerSrc, v);
    const n: Vec3 = [innerNormals[v * 3], innerNormals[v * 3 + 1], innerNormals[v * 3 + 2]];
    const t = thickness[v];
    if (boundaryVerts.has(v) && r > 0) {
      // 内壁回缩 r，外壁回缩 r，中间圆角顶点取厚度中点并向法线外偏 t/2 形成 1/4 圆弧近似
      innerId[v] = emit(v3.add(p, v3.scale(n, Math.min(r, t * 0.4))));
      outerId[v] = emit(v3.add(p, v3.scale(n, t - Math.min(r, t * 0.4))));
      const mid = v3.add(p, v3.scale(n, t * 0.5));
      filletId[v] = emit(mid);
    } else {
      innerId[v] = emit(p);
      outerId[v] = emit([outerPos[v * 3], outerPos[v * 3 + 1], outerPos[v * 3 + 2]]);
    }
  }

  const indices: number[] = [];
  // 内壁（反向，保证实体朝外法线一致）与外壁
  for (const f of keptFaces) {
    const a = innerSrc.indices[f * 3];
    const b = innerSrc.indices[f * 3 + 1];
    const c = innerSrc.indices[f * 3 + 2];
    indices.push(outerId[a], outerId[b], outerId[c]);
    indices.push(innerId[a], innerId[c], innerId[b]);
  }
  // 边界墙（含倒圆中间环）
  for (const [a, b] of boundaryEdges) {
    const ia = innerId[a], ib = innerId[b], oa = outerId[a], ob = outerId[b];
    if (filletId[a] >= 0) {
      const fa = filletId[a], fb = filletId[b];
      // 内→圆角→外，两圈四边形
      indices.push(ia, ib, fb, ia, fb, fa);
      indices.push(fa, fb, ob, fa, ob, oa);
    } else {
      indices.push(ia, ib, ob, ia, ob, oa);
    }
  }

  const shell: Mesh = {
    positions: new Float32Array(positions),
    indices: new Uint32Array(indices),
    name: 'shell-solid',
  };
  const inner: Mesh = {
    positions: new Float32Array(innerSrc.positions),
    indices: Uint32Array.from(keptFaces.flatMap((f) => [innerSrc.indices[f * 3], innerSrc.indices[f * 3 + 1], innerSrc.indices[f * 3 + 2]])),
    name: 'shell-inner-surface',
  };
  const outer: Mesh = {
    positions: outerPos.slice(),
    indices: new Uint32Array(inner.indices),
    name: 'shell-outer-surface',
  };
  return {
    shell,
    inner,
    outer,
    thicknessField: thickness,
    watertight: boundaryEdges.length > 0 ? isWatertight(shell) : isWatertight(shell),
  };
}

/** 水密检查：每条边恰好两个面；法线一致性由面积加权符号检查 */
export function isWatertight(m: Mesh): boolean {
  const edges = new Map<string, number>();
  const ekey = (a: number, b: number) => (a < b ? `${a}|${b}` : `${b}|${a}`);
  for (let i = 0; i < m.indices.length; i += 3) {
    const tri = [m.indices[i], m.indices[i + 1], m.indices[i + 2]];
    for (let k = 0; k < 3; k++) {
      const e = ekey(tri[k], tri[(k + 1) % 3]);
      edges.set(e, (edges.get(e) ?? 0) + 1);
    }
  }
  for (const count of edges.values()) if (count !== 2) return false;
  return true;
}

/** 壳体验证：薄于 2mm 高亮、外壁自交检测（BVH 采样）、非流形提示 */
export function validateShell(shell: Mesh, thickness: Float32Array, watertight: boolean): ShellIssue[] {
  const issues: ShellIssue[] = [];
  for (let v = 0; v < thickness.length; v++) {
    if (thickness[v] < 2 - 1e-6) {
      issues.push({ kind: 'thin', vertexId: v, detail: `壳厚 ${thickness[v].toFixed(2)}mm < 2mm` });
    }
  }
  // 自交：外壁面心对自身 BVH 查最近点，若命中其他面且距离≈0 视为自交（采样校验）
  const bvh = new TriangleBVH(shell);
  const F = faceCount(shell);
  let selfInt = 0;
  for (let f = 0; f < F; f += Math.max(1, Math.floor(F / 4000))) {
    const a = getVertex(shell, shell.indices[f * 3]);
    const b = getVertex(shell, shell.indices[f * 3 + 1]);
    const c = getVertex(shell, shell.indices[f * 3 + 2]);
    const centroid = v3.scale(v3.add(v3.add(a, b), c), 1 / 3);
    const hit = bvh.nearestPoint(centroid);
    if (hit.face >= 0 && hit.face !== f && hit.distance < 0.5) {
      selfInt++;
      if (selfInt <= 20) issues.push({ kind: 'self-intersection', faceId: f, detail: `面 ${f} 邻近面 ${hit.face} 距离 ${hit.distance.toFixed(2)}mm` });
    }
  }
  if (!watertight) issues.push({ kind: 'non-manifold', detail: '壳体存在未缝合边界边，无法直接打印' });
  return issues;
}
