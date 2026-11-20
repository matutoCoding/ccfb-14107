import { useEffect, useMemo, useState } from 'react';
import { useApp } from '../state/appState';
import { db, newId } from '../db/database';
import { Patient, Visit } from '../db/schema';

const EMPTY_PATIENT: Omit<Patient, 'id' | 'createdAt' | 'updatedAt'> = {
  name: '', gender: '男', birthDate: '', heightCm: 150, weightKg: 42, phone: '', note: '',
};

export function PatientsPage() {
  const { state, dispatch } = useApp();
  const [form, setForm] = useState(EMPTY_PATIENT);
  const [visitForm, setVisitForm] = useState({
    date: new Date().toISOString().slice(0, 10),
    cobbAngleDeg: 25, apexVertebra: 'T8', risserGrade: 2 as Visit['risserGrade'],
    heightCm: 150, weightKg: 42, curveDirection: '右凸' as Visit['curveDirection'], note: '',
  });

  useEffect(() => {
    (async () => {
      if (!state.currentPatient) {
        dispatch({ type: 'setVisits', visits: [] });
        dispatch({ type: 'setPlans', plans: [] });
        return;
      }
      dispatch({ type: 'setVisits', visits: await db.listVisits(state.currentPatient.id) });
      dispatch({ type: 'setPlans', plans: await db.listPlans(state.currentPatient.id) });
    })();
  }, [state.currentPatient]);

  const cobbPath = useMemo(() => buildCobbChart(state.visits), [state.visits]);

  async function savePatient() {
    if (!form.name.trim()) { dispatch({ type: 'toast', message: '请填写患者姓名' }); return; }
    const now = Date.now();
    const patient: Patient = { ...form, id: newId('pat'), createdAt: now, updatedAt: now };
    await db.put('patients', patient);
    const list = await db.getAll<Patient>('patients');
    list.sort((a, b) => b.updatedAt - a.updatedAt);
    dispatch({ type: 'setPatients', patients: list });
    dispatch({ type: 'setCurrentPatient', patient });
    setForm(EMPTY_PATIENT);
    dispatch({ type: 'toast', message: '患者已建档' });
  }

  async function addVisit() {
    if (!state.currentPatient) { dispatch({ type: 'toast', message: '请先选择患者' }); return; }
    const visitNo = state.visits.length + 1;
    const visit: Visit = {
      ...visitForm, id: newId('vis'), patientId: state.currentPatient.id, visitNo, createdAt: Date.now(),
    };
    await db.put('visits', visit);
    dispatch({ type: 'setVisits', visits: await db.listVisits(state.currentPatient.id) });
    dispatch({ type: 'toast', message: `第 ${visitNo} 次复诊已记录` });
  }

  async function removePatient(id: string) {
    if (!confirm('确认删除该患者及其全部复诊、扫描、方案？')) return;
    await db.deletePatientCascade(id);
    const list = await db.getAll<Patient>('patients');
    dispatch({ type: 'setPatients', patients: list });
    if (state.currentPatient?.id === id) dispatch({ type: 'setCurrentPatient' });
  }

  return (
    <div>
      <h2 className="page-title">患者档案</h2>
      <p className="page-sub">录入基本信息与脊柱指标，复诊角度变化自动绘制趋势；修型方案按复诊次数归档。</p>

      <div className="split">
        <div>
          <div className="card">
            <h3>新建患者</h3>
            <div className="form-row">
              <label>姓名</label>
              <input value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} />
            </div>
            <div className="grid-2">
              <div className="form-row">
                <label>性别</label>
                <select value={form.gender} onChange={(e) => setForm({ ...form, gender: e.target.value as Patient['gender'] })}>
                  <option>男</option><option>女</option>
                </select>
              </div>
              <div className="form-row">
                <label>出生日期</label>
                <input type="date" value={form.birthDate} onChange={(e) => setForm({ ...form, birthDate: e.target.value })} />
              </div>
            </div>
            <div className="grid-2">
              <div className="form-row">
                <label>身高 (cm)</label>
                <input type="number" value={form.heightCm} onChange={(e) => setForm({ ...form, heightCm: +e.target.value })} />
              </div>
              <div className="form-row">
                <label>体重 (kg)</label>
                <input type="number" value={form.weightKg} onChange={(e) => setForm({ ...form, weightKg: +e.target.value })} />
              </div>
            </div>
            <div className="form-row">
              <label>联系电话</label>
              <input value={form.phone} onChange={(e) => setForm({ ...form, phone: e.target.value })} />
            </div>
            <button className="btn primary" style={{ width: '100%' }} onClick={savePatient}>建立档案</button>
          </div>

          {state.currentPatient && (
            <div className="card">
              <h3>新增复诊记录</h3>
              <div className="grid-2">
                <div className="form-row"><label>复诊日期</label>
                  <input type="date" value={visitForm.date} onChange={(e) => setVisitForm({ ...visitForm, date: e.target.value })} /></div>
                <div className="form-row"><label>Cobb 角 (°)</label>
                  <input type="number" value={visitForm.cobbAngleDeg} onChange={(e) => setVisitForm({ ...visitForm, cobbAngleDeg: +e.target.value })} /></div>
                <div className="form-row"><label>顶椎节段</label>
                  <input value={visitForm.apexVertebra} onChange={(e) => setVisitForm({ ...visitForm, apexVertebra: e.target.value })} /></div>
                <div className="form-row"><label>Risser 分级</label>
                  <select value={visitForm.risserGrade} onChange={(e) => setVisitForm({ ...visitForm, risserGrade: +e.target.value as Visit['risserGrade'] })}>
                    {[0, 1, 2, 3, 4, 5].map((r) => <option key={r} value={r}>{r}</option>)}
                  </select></div>
                <div className="form-row"><label>身高 (cm)</label>
                  <input type="number" value={visitForm.heightCm} onChange={(e) => setVisitForm({ ...visitForm, heightCm: +e.target.value })} /></div>
                <div className="form-row"><label>体重 (kg)</label>
                  <input type="number" value={visitForm.weightKg} onChange={(e) => setVisitForm({ ...visitForm, weightKg: +e.target.value })} /></div>
              </div>
              <div className="form-row"><label>弯型方向</label>
                <select value={visitForm.curveDirection} onChange={(e) => setVisitForm({ ...visitForm, curveDirection: e.target.value as Visit['curveDirection'] })}>
                  <option>右凸</option><option>左凸</option><option>双弯</option>
                </select></div>
              <button className="btn primary" style={{ width: '100%' }} onClick={addVisit}>记录复诊</button>
            </div>
          )}
        </div>

        <div>
          <div className="card">
            <h3>患者列表</h3>
            <table>
              <thead><tr><th>姓名</th><th>性别</th><th>身高/体重</th><th>建档</th><th></th></tr></thead>
              <tbody>
                {state.patients.map((p) => (
                  <tr key={p.id} className="clickable" onClick={() => dispatch({ type: 'setCurrentPatient', patient: p })}
                    style={{ background: state.currentPatient?.id === p.id ? 'var(--primary-dim)' : undefined }}>
                    <td>{p.name}</td><td>{p.gender}</td>
                    <td>{p.heightCm}cm / {p.weightKg}kg</td>
                    <td>{new Date(p.createdAt).toLocaleDateString()}</td>
                    <td><button className="btn danger" onClick={(e) => { e.stopPropagation(); removePatient(p.id); }}>删除</button></td>
                  </tr>
                ))}
                {state.patients.length === 0 && <tr><td colSpan={5} className="muted">暂无患者，先在左侧建档</td></tr>}
              </tbody>
            </table>
          </div>

          {state.currentPatient && (
            <div className="card">
              <h3>Cobb 角复诊趋势 · {state.currentPatient.name}</h3>
              {state.visits.length >= 2 ? (
                <svg className="chart-line" viewBox="0 0 600 220">
                  <path d={cobbPath.line} fill="none" stroke="#1f6feb" strokeWidth="2.5" />
                  {cobbPath.points.map((p, i) => (
                    <g key={i}>
                      <circle cx={p.x} cy={p.y} r="4.5" fill="#1f6feb" />
                      <text x={p.x} y={p.y - 10} fontSize="12" textAnchor="middle">{p.cobb}°</text>
                      <text x={p.x} y="210" fontSize="11" textAnchor="middle" fill="#6b7686">第{p.no}次</text>
                    </g>
                  ))}
                </svg>
              ) : <p className="muted">至少两次复诊后绘制折线，当前 {state.visits.length} 次。</p>}
            </div>
          )}

          {state.currentPatient && (
            <div className="card">
              <h3>复诊与修型方案</h3>
              <table>
                <thead><tr><th>次数</th><th>日期</th><th>Cobb</th><th>顶椎</th><th>Risser</th><th>方案数</th></tr></thead>
                <tbody>
                  {state.visits.map((v) => (
                    <tr key={v.id}>
                      <td>第 {v.visitNo} 次</td><td>{v.date}</td>
                      <td>{v.cobbAngleDeg}°</td><td>{v.apexVertebra}</td><td>{v.risserGrade}</td>
                      <td>{state.plans.filter((pl) => pl.visitId === v.id).length}</td>
                    </tr>
                  ))}
                  {state.visits.length === 0 && <tr><td colSpan={6} className="muted">暂无复诊记录</td></tr>}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function buildCobbChart(visits: Visit[]) {
  const W = 600, H = 180, padX = 50;
  if (visits.length < 2) return { line: '', points: [] as { x: number; y: number; cobb: number; no: number }[] };
  const cobbs = visits.map((v) => v.cobbAngleDeg);
  const max = Math.max(...cobbs) * 1.15 || 10;
  const step = (W - padX * 2) / (visits.length - 1);
  const points = visits.map((v, i) => ({
    x: padX + step * i,
    y: H - 24 - (v.cobbAngleDeg / max) * (H - 50),
    cobb: v.cobbAngleDeg,
    no: v.visitNo,
  }));
  const line = points.map((p, i) => `${i === 0 ? 'M' : 'L'}${p.x},${p.y}`).join(' ');
  return { line, points };
}
