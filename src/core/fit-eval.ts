// 穿戴评估：壳体套回原始扫描，计算间隙/穿透/接触压力，超阈值点定位
import { TriangleMesh, SpatialHash } from './mesh';
import { Vec3, dot, sub } from './math';

export interface FitSample {
  vertex: number;
  /** 带符号间隙 mm：正=间隙，负=穿透 */
  gap: number;
  /** 估算接触压力 kPa（仅穿透点非零） */
  pressureKpa: number;
  position: Vec3;
  nearestFace: number;
}

export interface FitReport {
  samples: FitSample[];
  maxPenetrationMm: number;
  maxGapMm: number;
  meanGapMm: number;
  /** 超过皮肤耐受阈值的点 */
  overThreshold: FitSample[];
  contactAreaRatio: number;
}

/**
 * CPU 评估路径（空间哈希加速），与 GPU fit.wgsl 结果互为校验
 * @param pressureKpaPerMm 弹性基础标定系数，默认 12 kPa/mm
 */
export function evaluateFit(
  scan: TriangleMesh,
  shell: TriangleMesh,
  toleranceKpa: number,
  pressureKpaPerMm = 12,
): FitReport {
  const hash = new SpatialHash(shell, 10);
  const shellNormals = shell.vertexNormals();
  const samples: FitSample[] = [];
  let maxPen = 0;
  let maxGap = 0;
  let gapSum = 0;
  let contactCount = 0;

  for (let i = 0; i < scan.vertexCount; i++) {
    const q = scan.vertex(i);
    const hit = hash.nearest(q[0], q[1], q[2], 40);
    if (!hit) continue;
    // 符号：扫描点在壳面法向正侧为间隙
    const faceVerts = shell.face(hit.face);
    const anchor = shell.vertex(faceVerts[0]);
    const n: Vec3 = [
      shellNormals[faceVerts[0] * 3],
      shellNormals[faceVerts[0] * 3 + 1],
      shellNormals[faceVerts[0] * 3 + 2],
    ];
    const side = dot(sub(q, anchor), n);
    const gap = side >= 0 ? hit.distance : -hit.distance;
    const pressure = gap < 0 ? -gap * pressureKpaPerMm : 0;
    if (gap < 0) {
      maxPen = Math.max(maxPen, -gap);
      contactCount++;
    } else {
      maxGap = Math.max(maxGap, gap);
    }
    gapSum += gap;
    samples.push({ vertex: i, gap, pressureKpa: pressure, position: q, nearestFace: hit.face });
  }

  const overThreshold = samples
    .filter((s) => s.pressureKpa > toleranceKpa)
    .sort((a, b) => b.pressureKpa - a.pressureKpa);

  return {
    samples,
    maxPenetrationMm: maxPen,
    maxGapMm: maxGap,
    meanGapMm: gapSum / (samples.length || 1),
    overThreshold,
    contactAreaRatio: contactCount / (samples.length || 1),
  };
}

/** 把评估结果转成顶点标量场（间隙/压力），供色谱渲染 */
export function reportToVertexFields(scan: TriangleMesh, report: FitReport): {
  gap: Float32Array;
  pressure: Float32Array;
} {
  const gap = new Float32Array(scan.vertexCount);
  const pressure = new Float32Array(scan.vertexCount);
  for (const s of report.samples) {
    gap[s.vertex] = s.gap;
    pressure[s.vertex] = s.pressureKpa;
  }
  return { gap, pressure };
}
