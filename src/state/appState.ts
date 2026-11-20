import { createContext, useContext } from 'react';
import type { AppSettings, Patient, ScanRecord, SculptPlan, Visit } from '../db/schema';
import type { Mesh } from '../geometry/mesh';
import type { GpuCapability } from '../gpu/GpuCompute';

export type PageKey = 'patients' | 'scan' | 'sculpt' | 'shell' | 'fit' | 'settings';

/** 当前会话内活跃的网格（不进 React 深比较，单独持有） */
export interface ActiveMeshes {
  /** 原始（处理后摆正的）扫描 */
  scan?: Mesh;
  /** 修型工作副本 */
  sculpted?: Mesh;
  /** 修型前快照（差值对比用） */
  beforeSculpt?: Mesh;
  /** 壳体 */
  shell?: Mesh;
  /** 壳体内壁 */
  shellInner?: Mesh;
}

export interface AppState {
  page: PageKey;
  gpu: GpuCapability;
  settings: AppSettings;
  patients: Patient[];
  currentPatient?: Patient;
  visits: Visit[];
  currentVisit?: Visit;
  scans: ScanRecord[];
  currentScan?: ScanRecord;
  plans: SculptPlan[];
  currentPlan?: SculptPlan;
  meshes: ActiveMeshes;
  /** 全局提示消息 */
  toast?: string;
}

export const initialState: AppState = {
  page: 'patients',
  gpu: { available: false },
  settings: {
    id: 'singleton',
    skinToleranceKpa: 30,
    defaultShellThicknessMm: 4,
    printerMaxSize: { x: 300, y: 300, z: 400 },
    filletRadiusMm: 2,
    geodesicMethod: 'dijkstra',
    softTissueKpaPerMm: 8,
  },
  patients: [],
  visits: [],
  scans: [],
  plans: [],
  meshes: {},
};

export type Action =
  | { type: 'setPage'; page: PageKey }
  | { type: 'setGpu'; gpu: GpuCapability }
  | { type: 'setSettings'; settings: AppSettings }
  | { type: 'setPatients'; patients: Patient[] }
  | { type: 'setCurrentPatient'; patient?: Patient }
  | { type: 'setVisits'; visits: Visit[] }
  | { type: 'setCurrentVisit'; visit?: Visit }
  | { type: 'setScans'; scans: ScanRecord[] }
  | { type: 'setCurrentScan'; scan?: ScanRecord }
  | { type: 'setPlans'; plans: SculptPlan[] }
  | { type: 'setCurrentPlan'; plan?: SculptPlan }
  | { type: 'setMeshes'; meshes: ActiveMeshes }
  | { type: 'patchMesh'; key: keyof ActiveMeshes; mesh?: Mesh }
  | { type: 'toast'; message?: string };

export function reducer(state: AppState, action: Action): AppState {
  switch (action.type) {
    case 'setPage': return { ...state, page: action.page };
    case 'setGpu': return { ...state, gpu: action.gpu };
    case 'setSettings': return { ...state, settings: action.settings };
    case 'setPatients': return { ...state, patients: action.patients };
    case 'setCurrentPatient': return { ...state, currentPatient: action.patient };
    case 'setVisits': return { ...state, visits: action.visits };
    case 'setCurrentVisit': return { ...state, currentVisit: action.visit };
    case 'setScans': return { ...state, scans: action.scans };
    case 'setCurrentScan': return { ...state, currentScan: action.scan };
    case 'setPlans': return { ...state, plans: action.plans };
    case 'setCurrentPlan': return { ...state, currentPlan: action.plan };
    case 'setMeshes': return { ...state, meshes: action.meshes };
    case 'patchMesh': return { ...state, meshes: { ...state.meshes, [action.key]: action.mesh } };
    case 'toast': return { ...state, toast: action.message };
    default: return state;
  }
}

export const AppContext = createContext<{ state: AppState; dispatch: React.Dispatch<Action> } | null>(null);

export function useApp(): { state: AppState; dispatch: React.Dispatch<Action> } {
  const ctx = useContext(AppContext);
  if (!ctx) throw new Error('useApp 必须在 AppProvider 内使用');
  return ctx;
}
