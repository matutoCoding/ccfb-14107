// IndexedDB 仓储层：患者/扫描/方案/壳体/设置全部本地化存储，无后端
import {
  AppSettings,
  DEFAULT_SETTINGS,
  Patient,
  ScanRecord,
  SculptPlanRecord,
  ShellRecord,
} from './types';

const DB_NAME = 'scoli-brace-studio';
const DB_VERSION = 1;
const STORES = {
  patients: 'patients',
  scans: 'scans',
  plans: 'plans',
  shells: 'shells',
  settings: 'settings',
} as const;

let dbPromise: Promise<IDBDatabase> | null = null;

function openDB(): Promise<IDBDatabase> {
  if (dbPromise) return dbPromise;
  dbPromise = new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(STORES.patients)) {
        db.createObjectStore(STORES.patients, { keyPath: 'id' });
      }
      const scanStore = db.createObjectStore(STORES.scans, { keyPath: 'id' });
      scanStore.createIndex('patientId', 'patientId', { unique: false });
      const planStore = db.createObjectStore(STORES.plans, { keyPath: 'id' });
      planStore.createIndex('patientId', 'patientId', { unique: false });
      const shellStore = db.createObjectStore(STORES.shells, { keyPath: 'id' });
      shellStore.createIndex('planId', 'planId', { unique: false });
      db.createObjectStore(STORES.settings, { keyPath: 'id' });
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
  return dbPromise;
}

function tx<T>(store: string, mode: IDBTransactionMode, fn: (s: IDBObjectStore) => IDBRequest<T>): Promise<T> {
  return openDB().then(
    (db) =>
      new Promise<T>((resolve, reject) => {
        const t = db.transaction(store, mode);
        const req = fn(t.objectStore(store));
        req.onsuccess = () => resolve(req.result);
        req.onerror = () => reject(req.error);
      }),
  );
}

function txAll<T>(store: string, mode: IDBTransactionMode, fn: (s: IDBObjectStore) => IDBRequest<T[]>): Promise<T[]> {
  return openDB().then(
    (db) =>
      new Promise<T[]>((resolve, reject) => {
        const t = db.transaction(store, mode);
        const req = fn(t.objectStore(store));
        req.onsuccess = () => resolve(req.result);
        req.onerror = () => reject(req.error);
      }),
  );
}

export const repository = {
  // —— 患者 ——
  async savePatient(p: Patient): Promise<void> {
    await tx(STORES.patients, 'readwrite', (s) => s.put(p));
  },
  async getPatient(id: string): Promise<Patient | undefined> {
    return tx(STORES.patients, 'readonly', (s) => s.get(id));
  },
  async listPatients(): Promise<Patient[]> {
    const all = await txAll(STORES.patients, 'readonly', (s) => s.getAll());
    return all.sort((a, b) => b.createdAt - a.createdAt);
  },
  async deletePatient(id: string): Promise<void> {
    await tx(STORES.patients, 'readwrite', (s) => s.delete(id));
    // 级联删除关联数据
    const scans = await this.listScans(id);
    for (const sc of scans) await this.deleteScan(sc.id);
    const plans = await this.listPlans(id);
    for (const pl of plans) await this.deletePlan(pl.id);
  },

  // —— 扫描 ——
  async saveScan(rec: ScanRecord): Promise<void> {
    await tx(STORES.scans, 'readwrite', (s) => s.put(rec));
  },
  async getScan(id: string): Promise<ScanRecord | undefined> {
    return tx(STORES.scans, 'readonly', (s) => s.get(id));
  },
  async listScans(patientId?: string): Promise<ScanRecord[]> {
    if (!patientId) return txAll(STORES.scans, 'readonly', (s) => s.getAll());
    const db = await openDB();
    return new Promise((resolve, reject) => {
      const t = db.transaction(STORES.scans, 'readonly');
      const idx = t.objectStore(STORES.scans).index('patientId');
      const req = idx.getAll(patientId);
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => reject(req.error);
    });
  },
  async deleteScan(id: string): Promise<void> {
    await tx(STORES.scans, 'readwrite', (s) => s.delete(id));
  },

  // —— 修型方案 ——
  async savePlan(rec: SculptPlanRecord): Promise<void> {
    await tx(STORES.plans, 'readwrite', (s) => s.put(rec));
  },
  async getPlan(id: string): Promise<SculptPlanRecord | undefined> {
    return tx(STORES.plans, 'readonly', (s) => s.get(id));
  },
  async listPlans(patientId?: string): Promise<SculptPlanRecord[]> {
    if (!patientId) return txAll(STORES.plans, 'readonly', (s) => s.getAll());
    const db = await openDB();
    return new Promise((resolve, reject) => {
      const t = db.transaction(STORES.plans, 'readonly');
      const req = t.objectStore(STORES.plans).index('patientId').getAll(patientId);
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => reject(req.error);
    });
  },
  async deletePlan(id: string): Promise<void> {
    await tx(STORES.plans, 'readwrite', (s) => s.delete(id));
    const db = await openDB();
    await new Promise<void>((resolve, reject) => {
      const t = db.transaction(STORES.shells, 'readwrite');
      const idx = t.objectStore(STORES.shells).index('planId');
      const req = idx.getAllKeys(id);
      req.onsuccess = () => {
        const keys = req.result as string[];
        const store = t.objectStore(STORES.shells);
        keys.forEach((k) => store.delete(k));
        t.oncomplete = () => resolve();
      };
      req.onerror = () => reject(req.error);
    });
  },

  // —— 壳体 ——
  async saveShell(rec: ShellRecord): Promise<void> {
    await tx(STORES.shells, 'readwrite', (s) => s.put(rec));
  },
  async listShells(planId: string): Promise<ShellRecord[]> {
    const db = await openDB();
    return new Promise((resolve, reject) => {
      const t = db.transaction(STORES.shells, 'readonly');
      const req = t.objectStore(STORES.shells).index('planId').getAll(planId);
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => reject(req.error);
    });
  },
  async deleteShell(id: string): Promise<void> {
    await tx(STORES.shells, 'readwrite', (s) => s.delete(id));
  },

  // —— 设置 ——
  async getSettings(): Promise<AppSettings> {
    const s = await tx<AppSettings | undefined>(STORES.settings, 'readonly', (st) => st.get('singleton'));
    return { ...DEFAULT_SETTINGS, ...(s ?? {}) };
  },
  async saveSettings(s: AppSettings): Promise<void> {
    await tx(STORES.settings, 'readwrite', (st) => st.put(s));
  },

  async estimateStorage(): Promise<{ usage: number; quota: number }> {
    if (navigator.storage?.estimate) {
      const est = await navigator.storage.estimate();
      return { usage: est.usage ?? 0, quota: est.quota ?? 0 };
    }
    return { usage: 0, quota: 0 };
  },
};
