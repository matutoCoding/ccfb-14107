// 修型领域类型与纯函数（无 GPU 依赖，可在 Node 测试与 CPU 回退路径复用）
import { TriangleMesh } from './mesh';
import { Vec3 } from './math';

export type BrushKind = 'pressure' | 'relief';

export interface SculptDab {
  id: string;
  kind: BrushKind;
  /** 落笔顶点 */
  centerVertex: number;
  /** 测地衰减标准差 mm */
  sigmaMm: number;
  /** 推压/释放深度 mm（压垫为正，沿内法向） */
  depthMm: number;
  /** 对侧自动空腔（仅压垫） */
  contralateral?: {
    enabled: boolean;
    /** 空腔深度 mm（通常为压垫深度的 0.6~1 倍） */
    depthMm: number;
    sigmaScale: number;
  };
  label?: string;
}

export interface SculptPlan {
  id: string;
  patientId: string;
  visitIndex: number;
  dabs: SculptDab[];
  createdAt: number;
  /** 基线位置（方案应用前的原始扫描位置） */
  basePositions: Float32Array;
}

export interface SculptResult {
  positions: Float32Array;
  /** 每个 dab 的测地场（缓存，供色谱与后续编辑） */
  fields: Float32Array[];
  gpuMs: number;
}

export function defaultDab(centerVertex: number, kind: BrushKind): SculptDab {
  return {
    id: `dab-${Date.now()}-${Math.round(Math.random() * 1e4)}`,
    kind,
    centerVertex,
    sigmaMm: kind === 'pressure' ? 38 : 45,
    depthMm: kind === 'pressure' ? 12 : 6,
    contralateral: kind === 'pressure' ? { enabled: true, depthMm: 8, sigmaScale: 1.25 } : undefined,
  };
}

/** 沿脊柱高度（Y）抽取截面轮廓，供修型前后逐层对比 */
export interface CrossSection {
  y: number;
  before: Vec3[];
  after: Vec3[];
}

export function extractCrossSections(
  before: TriangleMesh,
  afterPositions: Float32Array,
  levels: number[],
  bandMm = 4,
): CrossSection[] {
  const sections: CrossSection[] = [];
  for (const y of levels) {
    const beforePts: Vec3[] = [];
    const afterPts: Vec3[] = [];
    const seen = new Set<number>();
    for (let f = 0; f < before.faceCount; f++) {
      const o = f * 3;
      for (let k = 0; k < 3; k++) {
        const vi = before.indices[o + k];
        if (seen.has(vi)) continue;
        const py = before.positions[vi * 3 + 1];
        if (Math.abs(py - y) <= bandMm) {
          seen.add(vi);
          beforePts.push(before.vertex(vi));
          afterPts.push([afterPositions[vi * 3], afterPositions[vi * 3 + 1], afterPositions[vi * 3 + 2]]);
        }
      }
    }
    const sortByAngle = (pts: Vec3[]) => {
      let cx = 0,
        cz = 0;
      pts.forEach((p) => ((cx += p[0]), (cz += p[2])));
      cx /= pts.length || 1;
      cz /= pts.length || 1;
      return pts
        .map((p) => ({ p, a: Math.atan2(p[2] - cz, p[0] - cx) }))
        .sort((u, v) => u.a - v.a)
        .map((e) => e.p);
    };
    sections.push({ y, before: sortByAngle(beforePts), after: sortByAngle(afterPts) });
  }
  return sections;
}
