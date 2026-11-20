import { useState } from 'react';
import { useApp } from '../state/appState';
import { db } from '../db/database';
import { AppSettings } from '../db/schema';

export function SettingsPage() {
  const { state, dispatch } = useApp();
  const [form, setForm] = useState<AppSettings>(state.settings);
  const [saved, setSaved] = useState(false);

  async function save() {
    await db.saveSettings(form);
    dispatch({ type: 'setSettings', settings: form });
    setSaved(true);
    dispatch({ type: 'toast', message: '设置已保存' });
    setTimeout(() => setSaved(false), 2000);
  }

  async function exportAll() {
    const data = {
      patients: await db.getAll('patients'),
      visits: await db.getAll('visits'),
      scans: await db.getAll('scans'),
      plans: await db.getAll('plans'),
      settings: await db.getSettings(),
      exportedAt: new Date().toISOString(),
    };
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `支具设计数据备份_${new Date().toISOString().slice(0, 10)}.json`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 4000);
  }

  async function clearAll() {
    if (!confirm('将清空本机全部患者、扫描网格与方案，且无法恢复。确认？')) return;
    const names = ['patients', 'visits', 'scans', 'plans', 'meshBlobs'] as const;
    for (const n of names) {
      const all = await db.getAll<{ id: string }>(n);
      for (const row of all) await db.delete(n, row.id);
    }
    dispatch({ type: 'setPatients', patients: [] });
    dispatch({ type: 'setMeshes', meshes: {} });
    dispatch({ type: 'toast', message: '本地数据已清空' });
  }

  return (
    <div>
      <h2 className="page-title">设置</h2>
      <p className="page-sub">临床阈值与制造参数。所有数据仅保存在本机浏览器 IndexedDB，不上传服务器。</p>
      <div className="grid-2">
        <div className="card">
          <h3>临床参数</h3>
          <div className="form-row">
            <label>皮肤耐受压力阈值（kPa）</label>
            <input type="number" min="5" max="80" value={form.skinToleranceKpa}
              onChange={(e) => setForm({ ...form, skinToleranceKpa: +e.target.value })} />
            <span className="muted" style={{ fontSize: 12 }}>超过该值的接触点会在穿戴评估中列出并预警</span>
          </div>
          <div className="form-row">
            <label>软组织等效刚度（kPa/mm）</label>
            <input type="number" min="1" max="20" step="0.5" value={form.softTissueKpaPerMm}
              onChange={(e) => setForm({ ...form, softTissueKpaPerMm: +e.target.value })} />
            <span className="muted" style={{ fontSize: 12 }}>接触压力 = 穿透深度 × 刚度（线性近似）</span>
          </div>
          <div className="form-row">
            <label>测地距离算法</label>
            <select value={form.geodesicMethod}
              onChange={(e) => setForm({ ...form, geodesicMethod: e.target.value as AppSettings['geodesicMethod'] })}>
              <option value="dijkstra">Dijkstra 图距离（稳健，推荐）</option>
              <option value="heat">Heat Method 热法（实验性）</option>
            </select>
          </div>
        </div>

        <div className="card">
          <h3>制造参数</h3>
          <div className="form-row">
            <label>默认壳厚（mm，范围 3–5）</label>
            <input type="number" min="3" max="5" step="0.5" value={form.defaultShellThicknessMm}
              onChange={(e) => setForm({ ...form, defaultShellThicknessMm: Math.min(5, Math.max(3, +e.target.value)) })} />
          </div>
          <div className="form-row">
            <label>默认切口倒圆半径（mm）</label>
            <input type="number" min="0" max="4" step="0.5" value={form.filletRadiusMm}
              onChange={(e) => setForm({ ...form, filletRadiusMm: +e.target.value })} />
          </div>
          <label className="form-row">
            <label>打印机最大成型尺寸（mm）</label>
            <div className="grid-3">
              <input type="number" value={form.printerMaxSize.x}
                onChange={(e) => setForm({ ...form, printerMaxSize: { ...form.printerMaxSize, x: +e.target.value } })} />
              <input type="number" value={form.printerMaxSize.y}
                onChange={(e) => setForm({ ...form, printerMaxSize: { ...form.printerMaxSize, y: +e.target.value } })} />
              <input type="number" value={form.printerMaxSize.z}
                onChange={(e) => setForm({ ...form, printerMaxSize: { ...form.printerMaxSize, z: +e.target.value } })} />
            </div>
            <span className="muted" style={{ fontSize: 12 }}>依次为 X / Y / Z；壳体超限时自动提示分段高度</span>
          </label>
        </div>

        <div className="card">
          <h3>计算引擎</h3>
          <div className="kv"><span className="k">WebGPU 通用计算</span>
            <span style={{ color: state.gpu.available ? 'var(--ok)' : 'var(--warn)' }}>
              {state.gpu.available ? '可用（WGSL compute）' : '不可用，CPU 兜底'}
            </span></div>
          <div className="kv"><span className="k">渲染</span><span>WebGL2 + SSAO</span></div>
          <div className="kv"><span className="k">本地存储</span><span>IndexedDB（元数据 + 网格二进制分表）</span></div>
        </div>

        <div className="card">
          <h3>数据管理</h3>
          <p className="muted" style={{ fontSize: 13 }}>导出 JSON 备份（不含网格二进制，网格可由 STL/OBJ 重新导入）。</p>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            <button className="btn" onClick={exportAll}>导出全部档案 JSON</button>
            <button className="btn danger" onClick={clearAll}>清空本机所有数据</button>
            <button className="btn primary" onClick={save}>{saved ? '已保存 ✓' : '保存设置'}</button>
          </div>
        </div>
      </div>
    </div>
  );
}
