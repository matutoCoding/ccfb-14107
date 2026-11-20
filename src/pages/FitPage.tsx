import { useMemo, useState } from 'react';
import { useApp } from '../state/appState';
import { Viewport, meshSummary } from '../components/Viewport';
import { Mesh, vertexCount } from '../geometry/mesh';
import { evaluateFit, pressureColormap, estimateCorrection, FitReport, PadInput } from '../geometry/fit';
import { serializeSTL, downloadBlob } from '../geometry/io';
import { isWatertight } from '../geometry/shell';
import { db, newId } from '../db/database';

export function FitPage() {
  const { state, dispatch } = useApp();
  const scan = state.meshes.scan;
  const shellInner = state.meshes.shellInner;
  const shell = state.meshes.shell;
  const [report, setReport] = useState<FitReport | null>(null);
  const [locateVertex, setLocateVertex] = useState<number | null>(null);

  const tolerance = state.settings.skinToleranceKpa;

  function runEvaluation() {
    if (!scan || !shellInner) { dispatch({ type: 'toast', message: '需要原始扫描与壳体内壁' }); return; }
    const r = evaluateFit(scan, shellInner, tolerance);
    setReport(r);
    dispatch({ type: 'toast', message: `评估完成：${r.violations.length} 个超阈值接触点` });
  }

  const coloredScan: Mesh | undefined = useMemo(() => {
    if (!scan || !report) return undefined;
    const colors = pressureColormap(report, vertexCount(scan), tolerance);
    return { ...scan, colors };
  }, [scan, report, tolerance]);

  // 矫正预测：从当前方案笔画构造压垫输入
  const correction = useMemo(() => {
    if (!scan || !state.currentPlan) return null;
    const pads: PadInput[] = state.currentPlan.strokes
      .filter((s) => s.kind === 'pad')
      .map((s) => {
        const v = s.centerVertex;
        const c: [number, number, number] = scan
          ? [scan.positions[v*3], scan.positions[v*3+1], scan.positions[v*3+2]]
          : [0, 0, 0];
        return { center: c, depthMm: s.depthMm, radiusMm: s.radiusMm, side: c[0] >= 0 ? 1 : -1 };
      });
    const baseCobb = state.visits[state.visits.length - 1]?.cobbAngleDeg ?? 25;
    return estimateCorrection(scan, baseCobb, pads);
  }, [scan, state.currentPlan, state.visits]);

  function exportSTL() {
    if (!shell) { dispatch({ type: 'toast', message: '尚无壳体' }); return; }
    if (!isWatertight(shell)) {
      if (!confirm('壳体未通过水密检查，仍要导出吗？')) return;
    }
    const buf = serializeSTL(shell);
    downloadBlob(buf, `${state.currentPatient?.name ?? 'brace'}_支具壳体.stl`, 'model/stl');
    dispatch({ type: 'toast', message: 'STL 已导出（二进制，毫米单位）' });
  }

  function exportWorkOrder() {
    if (!shell || !report || !correction) { dispatch({ type: 'toast', message: '请先完成壳体生成与穿戴评估' }); return; }
    const lines: string[] = [];
    lines.push('===== 色努式矫形支具修型单 =====');
    lines.push(`患者：${state.currentPatient?.name ?? '—'}    导出时间：${new Date().toLocaleString()}`);
    lines.push(`扫描：${state.currentScan?.name ?? '—'}（${state.currentScan?.vertexCount.toLocaleString()} 顶点）`);
    lines.push('');
    lines.push('一、压垫与释放区');
    state.currentPlan?.strokes.forEach((s, i) => {
      lines.push(`${i + 1}. [${s.kind === 'pad' ? '压垫' : '释放'}] 顶点#${s.centerVertex} 深度${s.depthMm}mm 半径${s.radiusMm}mm${s.contralateral ? `；对侧空腔 深度${s.contralateral.depthMm.toFixed(1)}mm` : ''}`);
    });
    lines.push('');
    lines.push('二、截面轮廓（沿高度采样，X/Z 坐标 mm）');
    const sculpted = state.meshes.sculpted;
    if (sculpted) {
      for (const y of [-250, -150, -50, 50, 150, 250]) {
        const ring: number[][] = [];
        for (let v = 0; v < vertexCount(sculpted); v++) {
          if (Math.abs(sculpted.positions[v*3+1] - y) <= 4) {
            ring.push([sculpted.positions[v*3], sculpted.positions[v*3+2]]);
          }
        }
        ring.sort((a, b) => Math.atan2(a[1], a[0]) - Math.atan2(b[1], b[0]));
        lines.push(`Y=${y}mm，${ring.length} 个轮廓点：`);
        lines.push(ring.slice(0, 24).map((p) => `(${p[0].toFixed(1)},${p[1].toFixed(1)})`).join(' '));
      }
    }
    lines.push('');
    lines.push('三、穿戴评估');
    lines.push(`最大穿透 ${report.maxPenetrationMm.toFixed(2)}mm；最大接触压力 ${report.maxPressureKpa.toFixed(1)}kPa；平均间隙 ${report.meanGapMm.toFixed(2)}mm；贴合率 ${(report.coverageRatio * 100).toFixed(1)}%`);
    lines.push(`皮肤耐受阈值 ${tolerance}kPa，超阈值点 ${report.violations.length} 个`);
    lines.push('');
    lines.push('四、矫正预测');
    lines.push(`原始 Cobb ${correction.originalCobb.toFixed(1)}° → 预估穿戴 ${correction.estimatedCobb.toFixed(1)}°，预估矫正率 ${(correction.correctionRate * 100).toFixed(1)}%`);
    lines.push('');
    lines.push('车间核对项：□ 水密 □ 法向一致 □ 壳厚分区 □ 后开口倒圆 □ 压垫对位 □ 无 <2mm 薄区');
    const blob = new Blob([lines.join('\n')], { type: 'text/plain;charset=utf-8' });
    downloadBlob(blob, `${state.currentPatient?.name ?? 'brace'}_修型单.txt`, 'text/plain');
  }

  async function finalize() {
    if (!state.currentPlan) { dispatch({ type: 'toast', message: '没有可定稿的修型方案' }); return; }
    const updated = { ...state.currentPlan, status: '定稿' as const, updatedAt: Date.now() };
    await db.put('plans', updated);
    dispatch({ type: 'setCurrentPlan', plan: updated });
    dispatch({ type: 'setPlans', plans: await db.listPlans(state.currentPatient!.id) });
    dispatch({ type: 'toast', message: '方案已定稿' });
  }

  return (
    <div>
      <h2 className="page-title">穿戴评估与定稿</h2>
      <p className="page-sub">壳体套回原始扫描，计算体表到内壁的有符号间隙与估算接触压力；超皮肤耐受阈值的点可点击定位。</p>
      <div className="split">
        <div>
          <div className="card">
            <h3>评估</h3>
            <button className="btn primary" style={{ width: '100%' }} onClick={runEvaluation} disabled={!scan || !shellInner}>
              运行间隙/穿透分析
            </button>
            {report && (
              <div style={{ marginTop: 14 }}>
                <div className="grid-2">
                  <Metric label="最大穿透" value={`${report.maxPenetrationMm.toFixed(1)}mm`} danger={report.maxPenetrationMm > 0} />
                  <Metric label="最大压力" value={`${report.maxPressureKpa.toFixed(0)}kPa`} danger={report.maxPressureKpa > tolerance} />
                  <Metric label="平均间隙" value={`${report.meanGapMm.toFixed(1)}mm`} />
                  <Metric label="贴合率" value={`${(report.coverageRatio * 100).toFixed(0)}%`} />
                </div>
              </div>
            )}
          </div>

          {correction && (
            <div className="card">
              <h3>矫正效果预测（正位）</h3>
              <CobbChart original={correction.originalCobb} estimated={correction.estimatedCobb} spine={correction.spineLine} />
              <div className="kv"><span className="k">原始 Cobb</span><span>{correction.originalCobb.toFixed(1)}°</span></div>
              <div className="kv"><span className="k">预估穿戴 Cobb</span><span>{correction.estimatedCobb.toFixed(1)}°</span></div>
              <div className="kv"><span className="k">预估矫正率</span>
                <span style={{ color: 'var(--ok)' }}>{(correction.correctionRate * 100).toFixed(1)}%</span></div>
              <p className="muted" style={{ fontSize: 12 }}>基于压垫位置/深度的力学简化估算，调整压垫后实时更新，非临床有限元结论。</p>
            </div>
          )}

          <div className="card">
            <h3>定稿与导出</h3>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              <button className="btn" onClick={exportSTL} disabled={!shell}>导出 STL（3D 打印）</button>
              <button className="btn" onClick={exportWorkOrder} disabled={!report}>导出修型单（截面/压垫清单）</button>
              <button className="btn primary" onClick={finalize} disabled={!state.currentPlan || state.currentPlan.status === '定稿'}>
                {state.currentPlan?.status === '定稿' ? '已定稿' : '定稿当前方案'}
              </button>
            </div>
          </div>
        </div>

        <div>
          <div className="card">
            <div className="toolbar">
              <strong>压力分布</strong>
              <span className="muted">{meshSummary(coloredScan ?? scan)}</span>
              {locateVertex !== null && <span className="tag pad">定位顶点 #{locateVertex}</span>}
            </div>
            {coloredScan ? (
              <Viewport specs={[
                { mesh: coloredScan, id: 'scan-pressure', alpha: 1, baseColor: [0.8, 0.85, 0.9], useVertexColor: true, visible: true },
                ...(shell ? [{ mesh: shell, id: 'shell-overlay', alpha: 0.3, baseColor: [0.78, 0.86, 0.98] as [number, number, number], visible: true }] : []),
              ]} />
            ) : <p className="muted">运行评估后，体表按接触压力上色（绿→黄→红）。</p>}
          </div>

          {report && report.violations.length > 0 && (
            <div className="card">
              <h3>超阈值接触点（点击定位）</h3>
              <div className="panel-scroll" style={{ maxHeight: 260 }}>
                <table>
                  <thead><tr><th>顶点</th><th>穿透 mm</th><th>压力 kPa</th></tr></thead>
                  <tbody>
                    {report.violations.slice(0, 100).map((fp) => (
                      <tr key={fp.vertexId} className="clickable" onClick={() => setLocateVertex(fp.vertexId)}>
                        <td>#{fp.vertexId}</td>
                        <td>{(-fp.gapMm).toFixed(2)}</td>
                        <td style={{ color: 'var(--danger)' }}>{fp.pressureKpa.toFixed(1)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function Metric({ label, value, danger }: { label: string; value: string; danger?: boolean }) {
  return (
    <div style={{ background: '#fafbfc', borderRadius: 8, padding: 12, marginBottom: 8 }}>
      <div className="metric-label">{label}</div>
      <div className="metric" style={{ color: danger ? 'var(--danger)' : 'var(--text)' }}>{value}</div>
    </div>
  );
}

function CobbChart({ original, estimated, spine }: { original: number; estimated: number; spine: [number, number, number][] }) {
  // 正位视图：X 横向偏移、Y 高度，画预估脊柱中线
  const xs = spine.map((p) => p[0]), ys = spine.map((p) => p[1]);
  const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys);
  const W = 300, H = 220, pad = 24;
  const px = (x: number) => pad + ((x - minX) / (maxX - minX || 1)) * (W - 2 * pad);
  const py = (y: number) => H - pad - ((y - minY) / (maxY - minY || 1)) * (H - 2 * pad);
  const path = spine.map((p, i) => `${i ? 'L' : 'M'}${px(p[0]).toFixed(1)},${py(p[1]).toFixed(1)}`).join(' ');
  return (
    <div>
      <svg width="100%" height={H} style={{ background: '#fafbfc', borderRadius: 8 }}>
        <line x1={W / 2} y1={pad} x2={W / 2} y2={H - pad} stroke="#dde3ea" strokeDasharray="4 4" />
        <path d={path} fill="none" stroke="#1f6feb" strokeWidth="2.5" />
      </svg>
      <p className="muted" style={{ fontSize: 12 }}>蓝线为预估脊柱中线；Cobb {original.toFixed(0)}° → {estimated.toFixed(0)}°</p>
    </div>
  );
}

void newId;
