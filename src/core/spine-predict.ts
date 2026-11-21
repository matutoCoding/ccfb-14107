// 矫正效果预测：按压垫位置/深度推断脊柱中线与预估 Cobb 角
import { TriangleMesh } from './mesh';
import { SculptDab } from './sculpt';
import { Vec3, deg, rad } from './math';

export interface SpinePrediction {
  /** 沿高度采样的预估脊柱中线（矫正后） */
  correctedCurve: Vec3[];
  originalCurve: Vec3[];
  /** 预估 Cobb 角（度） */
  predictedCobbDeg: number;
  originalCobbDeg: number;
  /** 矫正率 */
  correctionRate: number;
  /** 上下端椎角度的向量（正位视图绘制用） */
  endVectors: { superior: Vec3; inferior: Vec3 };
}

/**
 * 简化生物力学模型：
 * 1) 原始中线取网格各高度截面的质心 X 偏移
 * 2) 每个压垫在其高度产生 -depth * 效率系数 的横向矫正力矩
 * 3) 高斯影响沿高度扩散，叠加到原始曲线上得到矫正后曲线
 * 4) Cobb 角 = 上端椎/下端椎切线夹角
 */
export function predictSpine(
  mesh: TriangleMesh,
  dabs: SculptDab[],
  originalCobbDeg: number,
  levels = 40,
): SpinePrediction {
  const { min, max } = mesh.bbox();
  const height = max[1] - min[1];
  const pressureDabs = dabs.filter((d) => d.kind === 'pressure');

  // 原始中线：各高度带质心
  const originalCurve: Vec3[] = [];
  const correctedCurve: Vec3[] = [];
  const bandPoints: Vec3[][] = Array.from({ length: levels }, () => []);
  for (let i = 0; i < mesh.vertexCount; i++) {
    const y = mesh.positions[i * 3 + 1];
    const li = Math.min(levels - 1, Math.max(0, Math.floor(((y - min[1]) / height) * levels)));
    bandPoints[li].push(mesh.vertex(i));
  }
  for (let l = 0; l < levels; l++) {
    const pts = bandPoints[l];
    if (!pts.length) {
      originalCurve.push([0, min[1] + (l / levels) * height, 0]);
      continue;
    }
    let cx = 0,
      cy = 0,
      cz = 0;
    pts.forEach((p) => {
      cx += p[0];
      cy += p[1];
      cz += p[2];
    });
    cx /= pts.length;
    cy /= pts.length;
    cz /= pts.length;
    originalCurve.push([cx, cy, cz]);
  }

  // 压垫矫正贡献
  const correction = new Float32Array(levels);
  for (const dab of pressureDabs) {
    const center = mesh.vertex(dab.centerVertex);
    const centerLevel = ((center[1] - min[1]) / height) * levels;
    // 临床经验：压垫深度的横向矫正效率约 0.35，方向为压垫对侧（-x 方向若压垫在 +x）
    const side = center[0] >= 0 ? -1 : 1;
    const efficiency = 0.35;
    const sigmaLevel = (dab.sigmaMm / height) * levels * 1.4;
    for (let l = 0; l < levels; l++) {
      const w = Math.exp(-Math.pow((l - centerLevel) / sigmaLevel, 2));
      correction[l] += side * dab.depthMm * efficiency * w;
    }
  }
  for (let l = 0; l < levels; l++) {
    correction[l] = Math.max(-25, Math.min(25, correction[l]));
    correctedCurve.push([
      originalCurve[l][0] + correction[l],
      originalCurve[l][1],
      originalCurve[l][2],
    ]);
  }

  // Cobb 角：自动端椎选择——各层切线两两夹角的最大值（正位 X-Y 平面）
  const tangents = (curve: Vec3[]) => {
    const t: number[] = [];
    for (let l = 0; l < levels - 1; l++) {
      const dx = curve[l + 1][0] - curve[l][0];
      const dy = curve[l + 1][1] - curve[l][1];
      t.push(dx / Math.max(1e-6, dy));
    }
    return t;
  };
  const maxIncludedAngle = (slopes: number[]) => {
    let best = 0;
    for (let i = 0; i < slopes.length; i++) {
      for (let j = i + 1; j < slopes.length; j++) {
        // 两切线（slope, 1）的夹角
        const a = Math.atan(slopes[i]);
        const b = Math.atan(slopes[j]);
        best = Math.max(best, Math.abs(a - b));
      }
    }
    return deg(best);
  };
  const originalFromCurve = maxIncludedAngle(tangents(originalCurve));
  const predictedFromCurve = maxIncludedAngle(tangents(correctedCurve));
  // 用录入原始角做线性标定，保持与病例数据一致
  const scaleFactor = originalCobbDeg / Math.max(1e-3, originalFromCurve);
  const predictedCobbDeg = Math.max(0, predictedFromCurve * scaleFactor);
  const correctionRate = originalCobbDeg > 0 ? Math.max(0, 1 - predictedCobbDeg / originalCobbDeg) : 0;

  const tOrig = tangents(originalCurve);
  const tCorr = tangents(correctedCurve);
  return {
    correctedCurve,
    originalCurve,
    predictedCobbDeg,
    originalCobbDeg,
    correctionRate,
    endVectors: {
      superior: [tCorr[tCorr.length - 1] ?? 0, 1, 0],
      inferior: [tOrig[0] ?? 0, 1, 0],
    },
  };
}

/** 由端椎斜率生成正位视图折线（SVG 路径点） */
export function curveToPolyline(curve: Vec3[], width: number, height: number, bbox: { min: Vec3; max: Vec3 }): string {
  const rangeX = Math.max(
    40,
    ...curve.map((p) => Math.abs(p[0] - (bbox.min[0] + bbox.max[0]) / 2)),
  ) * 2;
  return curve
    .map((p, i) => {
      const x = width / 2 + ((p[0] - (bbox.min[0] + bbox.max[0]) / 2) / rangeX) * width * 0.8;
      const y = height - ((p[1] - bbox.min[1]) / (bbox.max[1] - bbox.min[1])) * height;
      return `${i === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(' ');
}

void rad;
