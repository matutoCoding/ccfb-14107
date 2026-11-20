/** 领域模型：患者、复诊、扫描、修型方案、设置 */

export interface Patient {
  id: string;
  name: string;
  gender: '男' | '女';
  birthDate: string; // ISO 日期
  heightCm: number;
  weightKg: number;
  /** 联系电话（可选） */
  phone?: string;
  note?: string;
  createdAt: number;
  updatedAt: number;
}

/** 一次复诊记录 */
export interface Visit {
  id: string;
  patientId: string;
  /** 第几次复诊（从 1 开始） */
  visitNo: number;
  date: string; // ISO 日期
  /** 主弯 Cobb 角（度） */
  cobbAngleDeg: number;
  /** 顶椎节段，如 T8 */
  apexVertebra: string;
  /** Risser 分级 0-5 */
  risserGrade: 0 | 1 | 2 | 3 | 4 | 5;
  heightCm: number;
  weightKg: number;
  /** 弯型方向 */
  curveDirection: '左凸' | '右凸' | '双弯';
  note?: string;
  createdAt: number;
}

/** 扫描记录（网格二进制单独存 Blob，这里只存元数据） */
export interface ScanRecord {
  id: string;
  patientId: string;
  visitId: string;
  name: string;
  source: 'obj' | 'ply' | 'synthetic';
  vertexCount: number;
  faceCount: number;
  /** 处理报告（补洞/清理统计） */
  processingReport?: {
    removedComponents: number;
    removedVertices: number;
    removedFaces: number;
    holesFilled: number;
    trianglesAdded: number;
  };
  /** 解剖标记点（摆正坐标系用） */
  landmarks: StoredLandmark[];
  createdAt: number;
}

export interface StoredLandmark {
  id: string;
  label: string;
  position: [number, number, number];
  vertexId: number;
}

/** 修型方案，按复诊挂在患者下 */
export interface SculptPlan {
  id: string;
  patientId: string;
  visitId: string;
  scanId: string;
  name: string;
  /** 笔刷笔画（压垫/释放区），几何可由参数重建 */
  strokes: StoredStroke[];
  /** 厚度分区 */
  thicknessZones: { yMin: number; yMax: number; thicknessMm: number }[];
  status: '草稿' | '定稿';
  createdAt: number;
  updatedAt: number;
}

export interface StoredStroke {
  id: string;
  kind: 'pad' | 'release';
  centerVertex: number;
  radiusMm: number;
  depthMm: number;
  label: string;
  createdAt: number;
  contralateral?: {
    centerVertex: number;
    radiusMm: number;
    depthMm: number;
  };
}

/** 应用设置 */
export interface AppSettings {
  id: 'singleton';
  /** 皮肤耐受压力阈值（kPa） */
  skinToleranceKpa: number;
  /** 默认壳厚（mm） */
  defaultShellThicknessMm: number;
  /** 打印机最大成型尺寸（mm） */
  printerMaxSize: { x: number; y: number; z: number };
  /** 倒圆半径（mm） */
  filletRadiusMm: number;
  /** 测地算法 */
  geodesicMethod: 'dijkstra' | 'heat';
  /** 软组织等效刚度 kPa/mm */
  softTissueKpaPerMm: number;
}

export const DEFAULT_SETTINGS: AppSettings = {
  id: 'singleton',
  skinToleranceKpa: 30,
  defaultShellThicknessMm: 4,
  printerMaxSize: { x: 300, y: 300, z: 400 },
  filletRadiusMm: 2,
  geodesicMethod: 'dijkstra',
  softTissueKpaPerMm: 8,
};

/** 大块网格二进制（OBJ/PLY 原始字节 + 处理后的顶点/索引），与元数据分表 */
export interface MeshBlob {
  id: string; // = scanId
  /** 处理后的位置 Float32 字节 */
  positions: ArrayBuffer;
  /** 处理后的索引 Uint32 字节 */
  indices: ArrayBuffer;
  /** 原始上传文件（可选，留档） */
  original?: Blob;
}

export type StoreName = 'patients' | 'visits' | 'scans' | 'plans' | 'settings' | 'meshBlobs';
