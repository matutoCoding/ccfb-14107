// 领域模型：患者、复诊、扫描、修型方案、壳体、设置
import { SculptDab } from './sculpt';
import { ThicknessZone, WindowCut, TrimLine } from './shell';

export interface CobbVisit {
  id: string;
  date: string;          // ISO 日期
  visitIndex: number;
  cobbAngleDeg: number;
  apexVertebra: string;  // 顶椎节段，如 T8
  risserGrade: number;   // 0-5
  heightCm: number;
  weightKg: number;
  note?: string;
}

export interface Patient {
  id: string;
  name: string;
  gender: '男' | '女';
  birthDate: string;
  contact?: string;
  diagnosis?: string;
  createdAt: number;
  visits: CobbVisit[];
}

export interface ScanRecord {
  id: string;
  patientId: string;
  visitIndex: number;
  filename: string;
  createdAt: number;
  /** 网格几何存入 IndexedDB（二进制） */
  positions: Float32Array;
  indices: Uint32Array;
  /** 解剖标记点（mm，解剖坐标系摆正后） */
  landmarks: {
    asiLeft: [number, number, number];
    asiRight: [number, number, number];
    axillaLeft: [number, number, number];
    axillaRight: [number, number, number];
  };
  /** 处理统计 */
  cleaningStats?: {
    removedComponents: number;
    removedFaces: number;
    filledHoles: number;
    filledTriangles: number;
  };
  synthetic?: boolean;
}

export interface SculptPlanRecord {
  id: string;
  patientId: string;
  visitIndex: number;
  scanId: string;
  name: string;
  createdAt: number;
  dabs: SculptDab[];
  /** 修型后位置（定稿时固化） */
  sculptedPositions?: Float32Array;
  basePositions: Float32Array;
}

export interface ShellRecord {
  id: string;
  planId: string;
  patientId: string;
  createdAt: number;
  positions: Float32Array;
  indices: Uint32Array;
  thicknessField: Float32Array;
  thicknessZones: ThicknessZone[];
  trimLines: TrimLine[];
  windows: WindowCut[];
  posteriorOpeningMm: number;
  validation?: {
    watertight: boolean;
    minThicknessMm: number;
    maxThicknessMm: number;
    thicknessErrorMax: number;
    thinVertexCount: number;
    selfIntersectCount: number;
  };
}

export interface AppSettings {
  id: 'singleton';
  /** 皮肤耐受压力阈值 kPa */
  skinToleranceKpa: number;
  /** 默认壳厚 mm */
  defaultShellThicknessMm: number;
  /** 打印机最大成型尺寸 mm */
  printerMaxSize: { x: number; y: number; z: number };
  /** 压力标定 kPa/mm */
  pressureCoeffKpaPerMm: number;
  /** CG 迭代数 */
  cgIterations: number;
}

export const DEFAULT_SETTINGS: AppSettings = {
  id: 'singleton',
  skinToleranceKpa: 26.7, // 约 200 mmHg 量级的临床常用耐受阈值（kPa）
  defaultShellThicknessMm: 4,
  printerMaxSize: { x: 300, y: 300, z: 400 },
  pressureCoeffKpaPerMm: 12,
  cgIterations: 120,
};

export function uid(prefix = 'id'): string {
  return `${prefix}-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
}
