import { Mesh, vertexCount, getVertex } from './mesh';
import { v3 } from './math';
import { TriangleBVH } from './bvh';

export interface FitPoint {
  vertexId: number;
  /** 间隙（毫米，正=壳内壁与皮肤间有空隙）；负值=穿透 */
  gapMm: number;
  /** 估算接触压力 kPa（仅穿透点 >0） */
  pressureKpa: number;
  position: [number, number, number];
}

export interface FitReport {
  points: FitPoint[];
  violations: FitPoint[];
  maxPenetrationMm: number;
  maxPressureKpa: number;
  meanGapMm: number;
  coverageRatio: number; // |gap| <= 3mm 的贴合占比
}

/**
 * 软组织等效线性刚度（简化模型）：
 * 躯干软组织受压近似 p = k·δ，k 取 8 kPa/mm（文献 5–12 的中位，
 * 真实评估应按部位分区，设置页可替换全局阈值）。
 */
const SOFT_TISSUE_K = 8;

/**
 * 把壳体套回原始扫描：对原始扫描每个顶点查壳体内壁 BVH，
 * 用有符号法线距离判定——扫描点相对命中面法线：
 *   (p - hitPoint)·N > 0：点在壳外（皮肤与内壁间有间隙，正）
 *   (p - hitPoint)·N < 0：点侵入壳材料内部（穿透，负）
 */
export function evaluateFit(originalScan: Mesh, shellInner: Mesh, toleranceKpa: number): FitReport {
  const bvh = new TriangleBVH(shellInner);
  const V = vertexCount(originalScan);
  const points: FitPoint[] = new Array(V);
  let maxPen = 0, maxP = 0, gapSum = 0, fitCount = 0;
  const violations: FitPoint[] = [];

  for (let v = 0; v < V; v++) {
    const p = getVertex(originalScan, v);
    const hit = bvh.nearestPoint(p);
    // 有符号距离：法线方向为正（壳外），反之为负（穿透）
    const signedGap = v3.dot(v3.sub(p, hit.point), hit.normal);
    const pressure = signedGap < 0 ? -signedGap * SOFT_TISSUE_K : 0;
    const fp: FitPoint = {
      vertexId: v,
      gapMm: signedGap,
      pressureKpa: pressure,
      position: [p[0], p[1], p[2]],
    };
    points[v] = fp;
    if (signedGap < 0) {
      maxPen = Math.max(maxPen, -signedGap);
      maxP = Math.max(maxP, pressure);
      if (pressure > toleranceKpa) violations.push(fp);
    }
    gapSum += signedGap;
    if (Math.abs(signedGap) <= 3) fitCount++;
  }
  violations.sort((a, b) => b.pressureKpa - a.pressureKpa);
  return {
    points,
    violations,
    maxPenetrationMm: maxPen,
    maxPressureKpa: maxP,
    meanGapMm: gapSum / V,
    coverageRatio: fitCount / V,
  };
}

/** 压力场 → 顶点色（绿→黄→红，超阈值闪红） */
export function pressureColormap(report: FitReport, V: number, toleranceKpa: number): Float32Array {
  const colors = new Float32Array(V * 3);
  for (const fp of report.points) {
    const i = fp.vertexId * 3;
    if (fp.gapMm >= 0) {
      const t = Math.min(1, fp.gapMm / 15);
      colors[i] = 0.75 - t * 0.35;
      colors[i + 1] = 0.85 - t * 0.2;
      colors[i + 2] = 1 - t * 0.4;
    } else {
      const t = Math.min(1, fp.pressureKpa / Math.max(toleranceKpa, 1));
      if (t < 0.6) {
        colors[i] = 0.4 + t; colors[i + 1] = 0.8; colors[i + 2] = 0.2;
      } else {
        colors[i] = 0.95; colors[i + 1] = 0.35 * (1 - (t - 0.6) / 0.4); colors[i + 2] = 0.15;
      }
    }
  }
  return colors;
}

/** 压垫信息（矫正预测输入） */
export interface PadInput {
  center: [number, number, number];
  depthMm: number;
  radiusMm: number;
  /** 凸侧：+1 患者左侧凸（+X），-1 右侧凸 */
  side: number;
}

export interface CorrectionEstimate {
  /** 预估脊柱中线（沿 Y 的折线点） */
  spineLine: [number, number, number][];
  /** 原始/预估 Cobb 角（度），上下端椎分别拟合 */
  originalCobb: number;
  estimatedCobb: number;
  correctionRate: number;
  apexY: number;
}

/**
 * 矫正效果估算（力学简化模型，非临床有限元）：
 * 压垫在凸侧施加横向推压，脊柱中线按深度的 35%（文献报道支具即刻矫正率
 * 通常 40–60%，取保守端）向凹侧平移，影响沿高度按测地式高斯衰减。
 * 上下端椎方向用最小二乘拟合后算夹角得 Cobb 角。
 */
export function estimateCorrection(
  scan: Mesh,
  originalCobbDeg: number,
  pads: PadInput[],
): CorrectionEstimate {
  // 从扫描各高度截面的几何中心提取原始脊柱中线代理
  const bins = 32;
  let yMin = Infinity, yMax = -Infinity;
  for (let i = 1; i < scan.positions.length; i += 3) {
    yMin = Math.min(yMin, scan.positions[i]);
    yMax = Math.max(yMax, scan.positions[i]);
  }
  const sums = Array.from({ length: bins }, () => ({ x: 0, z: 0, y: 0, n: 0 }));
  for (let v = 0; v < vertexCount(scan); v++) {
    const p = getVertex(scan, v);
    const b = Math.min(bins - 1, Math.max(0, Math.floor(((p[1] - yMin) / (yMax - yMin)) * bins)));
    sums[b].x += p[0]; sums[b].z += p[2]; sums[b].y += p[1]; sums[b].n++;
  }
  const rawLine: [number, number, number][] = sums
    .filter((s) => s.n > 0)
    .map((s) => [s.x / s.n, s.y / s.n, s.z / s.n]);

  const corrected = rawLine.map(([x, y, z]) => {
    let dx = 0;
    for (const pad of pads) {
      const dy = (y - pad.center[1]) / pad.radiusMm;
      const w = Math.exp(-dy *dy / 2) * (pad.depthMm * 0.35);
      dx += -pad.side * w;
    }
    return [x + dx, y, z] as [number, number, number];
  });

  const cobbFromLine = (line: [number, number, number][]): { cobb: number; apexY: number } => {
    // 顶椎 = |x| 相对中线偏移最大处；上下端椎取两端 1/3 段做线性拟合
    const meanX = line.reduce((s, p) => s + p[0], 0) / line.length;
    let apex = line[0];
    for (const p of line) if (Math.abs(p[0] - meanX) > Math.abs(apex[0] - meanX)) apex = p;
    const third = Math.max(2, Math.floor(line.length / 3));
    const slope = (seg: [number, number, number][]) => {
      const my = seg.reduce((s, p) => s + p[1], 0) / seg.length;
      const mx = seg.reduce((s, p) => s + p[0], 0) / seg.length;
      let num = 0, den = 0;
      for (const p of seg) { num += (p[1] - my) * (p[0] - mx); den += (p[1] - my) ** 2; }
      return num / (den || 1);
    };
    const sTop = slope(line.slice(0, third));
    const sBot = slope(line.slice(line.length - third));
    const ang = (m: number) => Math.atan(m) * 180 / Math.PI;
    return { cobb: Math.abs(ang(sTop) - ang(sBot)), apexY: apex[1] };
  };

  const orig = cobbFromLine(rawLine);
  const est = cobbFromLine(corrected);
  // 扫描几何推导的 Cobb 仅作形状参考；绝对角度以病历录入值为基准按比例缩放
  const scale = orig.cobb > 1e-6 ? originalCobbDeg / orig.cobb : 1;
  const estimatedCobb = est.cobb * scale;
  return {
    spineLine: corrected,
    originalCobb: originalCobbDeg,
    estimatedCobb: Math.max(0, estimatedCobb),
    correctionRate: originalCobbDeg > 0 ? Math.max(0, 1 - estimatedCobb / originalCobbDeg) : 0,
    apexY: orig.apexY,
  };
}

/** 打印机成型尺寸分段提示 */
export function splitSuggestion(
  shell: Mesh,
  maxBuild: { x: number; y: number; z: number },
): { fits: boolean; overAxis?: 'x' | 'y' | 'z'; cutHeights: number[]; shellHeight: number } {
  let min: [number, number, number] = [Infinity, Infinity, Infinity];
  let max: [number, number, number] = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < shell.positions.length; i += 3) {
    for (let k = 0; k < 3; k++) {
      min[k] = Math.min(min[k], shell.positions[i + k]);
      max[k] = Math.max(max[k], shell.positions[i + k]);
    }
  }
  const size: [number, number, number] = [max[0] - min[0], max[1] - min[1], max[2] - min[2]];
  const axes: ('x' | 'y' | 'z')[] = ['x', 'y', 'z'];
  const limits = [maxBuild.x, maxBuild.y, maxBuild.z];
  let over: 'x' | 'y' | 'z' | undefined;
  for (let k = 0; k < 3; k++) if (size[k] > limits[k]) { over = axes[k]; break; }
  const cutHeights: number[] = [];
  if (over === 'y') {
    const pieces = Math.ceil(size[1] / maxBuild.y);
    for (let i = 1; i < pieces; i++) cutHeights.push(min[1] + (size[1] / pieces) * i);
  }
  return { fits: !over, overAxis: over, cutHeights, shellHeight: size[1] };
}

void v3;
