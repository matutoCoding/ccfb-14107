import {
  AppSettings, DEFAULT_SETTINGS, MeshBlob, Patient, ScanRecord, SculptPlan, StoreName, Visit,
} from './schema';

const DB_NAME = 'chaneau-brace-studio';
const DB_VERSION = 1;

/**
 * IndexedDB 封装。所有患者数据与网格二进制均留在浏览器本地，
 * 不上传任何服务器。网格 Blob 单独成表，避免拉元数据时载入大数组。
 */
class BraceDatabase {
  private db: IDBDatabase | null = null;

  async open(): Promise<IDBDatabase> {
    if (this.db) return this.db;
    return new Promise((resolve, reject) => {
      const req = indexedDB.open(DB_NAME, DB_VERSION);
      req.onupgradeneeded = () => {
        const db = req.result;
        if (!db.objectStoreNames.contains('patients')) {
          db.createObjectStore('patients', { keyPath: 'id' });
        }
        if (!db.objectStoreNames.contains('visits')) {
          const s = db.createObjectStore('visits', { keyPath: 'id' });
          s.createIndex('patientId', 'patientId', { unique: false });
        }
        if (!db.objectStoreNames.contains('scans')) {
          const s = db.createObjectStore('scans', { keyPath: 'id' });
          s.createIndex('patientId', 'patientId', { unique: false });
          s.createIndex('visitId', 'visitId', { unique: false });
        }
        if (!db.objectStoreNames.contains('plans')) {
          const s = db.createObjectStore('plans', { keyPath: 'id' });
          s.createIndex('patientId', 'patientId', { unique: false });
          s.createIndex('scanId', 'scanId', { unique: false });
        }
        if (!db.objectStoreNames.contains('settings')) {
          db.createObjectStore('settings', { keyPath: 'id' });
        }
        if (!db.objectStoreNames.contains('meshBlobs')) {
          db.createObjectStore('meshBlobs', { keyPath: 'id' });
        }
      };
      req.onsuccess = () => { this.db = req.result; resolve(req.result); };
      req.onerror = () => reject(req.error);
    });
  }

  private async tx<T>(store: StoreName, mode: IDBTransactionMode, fn: (s: IDBObjectStore) => IDBRequest<T>): Promise<T> {
    const db = await this.open();
    return new Promise((resolve, reject) => {
      const t = db.transaction(store, mode);
      const req = fn(t.objectStore(store));
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => reject(req.error);
    });
  }

  async put<T>(store: StoreName, value: T): Promise<IDBValidKey> {
    return this.tx(store, 'readwrite', (s) => s.put(value as unknown as Record<string, unknown>));
  }

  async get<T>(store: StoreName, key: string): Promise<T | undefined> {
    return this.tx<T | undefined>(store, 'readonly', (s) => s.get(key) as IDBRequest<T | undefined>);
  }

  async getAll<T>(store: StoreName): Promise<T[]> {
    return this.tx<T[]>(store, 'readonly', (s) => s.getAll() as IDBRequest<T[]>);
  }

  async delete(store: StoreName, key: string): Promise<void> {
    await this.tx(store, 'readwrite', (s) => s.delete(key));
  }

  async queryByIndex<T>(store: StoreName, index: string, value: string): Promise<T[]> {
    const db = await this.open();
    return new Promise((resolve, reject) => {
      const t = db.transaction(store, 'readonly');
      const req = t.objectStore(store).index(index).getAll(value);
      req.onsuccess = () => resolve(req.result as T[]);
      req.onerror = () => reject(req.error);
    });
  }

  // ---- 业务便捷方法 ----
  async getSettings(): Promise<AppSettings> {
    const s = await this.get<AppSettings>('settings', 'singleton');
    return s ?? DEFAULT_SETTINGS;
  }

  async saveSettings(s: AppSettings): Promise<void> {
    await this.put('settings', s);
  }

  async listVisits(patientId: string): Promise<Visit[]> {
    const list = await this.queryByIndex<Visit>('visits', 'patientId', patientId);
    return list.sort((a, b) => a.visitNo - b.visitNo);
  }

  async listScans(patientId: string): Promise<ScanRecord[]> {
    const list = await this.queryByIndex<ScanRecord>('scans', 'patientId', patientId);
    return list.sort((a, b) => b.createdAt - a.createdAt);
  }

  async listPlans(patientId: string): Promise<SculptPlan[]> {
    const list = await this.queryByIndex<SculptPlan>('plans', 'patientId', patientId);
    return list.sort((a, b) => b.updatedAt - a.updatedAt);
  }

  async saveMeshBlob(blob: MeshBlob): Promise<void> {
    await this.put('meshBlobs', blob);
  }

  async getMeshBlob(scanId: string): Promise<MeshBlob | undefined> {
    return this.get<MeshBlob>('meshBlobs', scanId);
  }

  /** 删除患者的级联数据 */
  async deletePatientCascade(patientId: string): Promise<void> {
    const db = await this.open();
    const visits = await this.listVisits(patientId);
    const scans = await this.listScans(patientId);
    const plans = await this.listPlans(patientId);
    await new Promise<void>((resolve, reject) => {
      const t = db.transaction(['visits', 'scans', 'plans', 'meshBlobs', 'patients'], 'readwrite');
      t.oncomplete = () => resolve();
      t.onerror = () => reject(t.error);
      for (const v of visits) t.objectStore('visits').delete(v.id);
      for (const sc of scans) {
        t.objectStore('scans').delete(sc.id);
        t.objectStore('meshBlobs').delete(sc.id);
      }
      for (const p of plans) t.objectStore('plans').delete(p.id);
      t.objectStore('patients').delete(patientId);
    });
  }
}

export const db = new BraceDatabase();
export function newId(prefix: string): string {
  return `${prefix}_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`;
}
