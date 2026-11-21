// 全局应用状态：当前患者/扫描/方案/壳体 + 订阅通知（无框架轻量实现）
import { repository } from '../core/repository';
import { AppSettings, Patient, ScanRecord, SculptPlanRecord, ShellRecord, uid } from '../core/types';
import { TriangleMesh } from '../core/mesh';

export interface SessionState {
  settings: AppSettings;
  patients: Patient[];
  currentPatient: Patient | null;
  currentScan: ScanRecord | null;
  currentPlan: SculptPlanRecord | null;
  currentShell: ShellRecord | null;
  /** 运行期网格（不持久化到 state，由会话缓存管理） */
  meshVersion: number;
}

type Listener = (state: SessionState) => void;

class Store {
  state: SessionState = {
    settings: {
      id: 'singleton',
      skinToleranceKpa: 26.7,
      defaultShellThicknessMm: 4,
      printerMaxSize: { x: 300, y: 300, z: 400 },
      pressureCoeffKpaPerMm: 12,
      cgIterations: 120,
    },
    patients: [],
    currentPatient: null,
    currentScan: null,
    currentPlan: null,
    currentShell: null,
    meshVersion: 0,
  };
  private listeners = new Set<Listener>();

  subscribe(fn: Listener): () => void {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  }

  private emit() {
    this.state = { ...this.state };
    this.listeners.forEach((fn) => fn(this.state));
  }

  async init() {
    this.state.settings = await repository.getSettings();
    this.state.patients = await repository.listPatients();
    this.emit();
  }

  async saveSettings(patch: Partial<AppSettings>) {
    this.state.settings = { ...this.state.settings, ...patch };
    await repository.saveSettings(this.state.settings);
    this.emit();
  }

  async savePatient(p: Patient) {
    await repository.savePatient(p);
    this.state.patients = await repository.listPatients();
    if (this.state.currentPatient?.id === p.id) this.state.currentPatient = p;
    this.emit();
  }

  async deletePatient(id: string) {
    await repository.deletePatient(id);
    this.state.patients = await repository.listPatients();
    if (this.state.currentPatient?.id === id) {
      this.state.currentPatient = null;
      this.state.currentScan = null;
      this.state.currentPlan = null;
      this.state.currentShell = null;
    }
    this.emit();
  }

  selectPatient(p: Patient | null) {
    this.state.currentPatient = p;
    this.state.currentScan = null;
    this.state.currentPlan = null;
    this.state.currentShell = null;
    this.emit();
  }

  async selectScan(scan: ScanRecord | null) {
    this.state.currentScan = scan;
    this.state.currentPlan = null;
    this.state.currentShell = null;
    this.state.meshVersion++;
    this.emit();
  }

  async saveScan(scan: ScanRecord) {
    await repository.saveScan(scan);
    this.state.currentScan = scan;
    this.state.meshVersion++;
    this.emit();
  }

  async listScans(patientId: string) {
    return repository.listScans(patientId);
  }

  async savePlan(plan: SculptPlanRecord) {
    await repository.savePlan(plan);
    this.state.currentPlan = plan;
    this.emit();
  }

  selectPlan(plan: SculptPlanRecord | null) {
    this.state.currentPlan = plan;
    this.state.currentShell = null;
    this.emit();
  }

  async saveShell(shell: ShellRecord) {
    await repository.saveShell(shell);
    this.state.currentShell = shell;
    this.emit();
  }

  selectShell(shell: ShellRecord | null) {
    this.state.currentShell = shell;
    this.emit();
  }

  bumpMesh() {
    this.state.meshVersion++;
    this.emit();
  }

  newId(prefix: string) {
    return uid(prefix);
  }
}

export const store = new Store();

/** ScanRecord -> TriangleMesh */
export function scanToMesh(scan: ScanRecord): TriangleMesh {
  return new TriangleMesh(new Float32Array(scan.positions), new Uint32Array(scan.indices));
}

export function shellToMesh(shell: ShellRecord): TriangleMesh {
  return new TriangleMesh(new Float32Array(shell.positions), new Uint32Array(shell.indices));
}
