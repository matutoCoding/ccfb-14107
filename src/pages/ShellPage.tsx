import { useMemo, useState } from 'react';
import { useApp } from '../state/appState';
import { Viewport, meshSummary } from '../components/Viewport';
import { Mesh, meshBounds } from '../geometry/mesh';
import { buildShell, ThicknessZone, ShellIssue, BackOpening } from '../geometry/shell';
import { splitSuggestion } from '../geometry/fit';

export function ShellPage() {
  const { state, dispatch } = useApp();
  const sculpted = state.meshes.sculpted ?? state.meshes.scan;
  const [zones, setZones] = useState<ThicknessZone[]>([
    { yMin: -360, yMax: -80, thicknessMm: 4 },
    { yMin: -80, yMax: 120, thicknessMm: 5 },
    { yMin: 120, yMax: 360, thicknessMm: 3 },
  ]);
  const [petg, setPetg] = useState(true);
  const [fillet, setFillet] = useState(state.settings.filletRadiusMm);
  const [backOpen, setBackOpen] = useState(true);
  const [issues, setIssues] = useState<ShellIssue[]>([]);
  const [watertight, setWatertight] = useState<boolean | null>(null);
  const [split, setSplit] = useState<ReturnType<typeof splitSuggestion> | null>(null);

  const bounds = useMemo(() => (sculpted ? meshBounds(sculpted) : null), [sculpted]);

  function addZone() {
    setZones((z) => [...z, { yMin: 0, yMax: 100, thicknessMm: state.settings.defaultShellThicknessMm }]);
  }

  function generate() {
    if (!sculpted) { dispatch({ type: 'toast', message: '请先完成扫描与修型' }); return; }
    const bo: BackOpening | undefined = backOpen
      ? { halfAngle: 0.22, yMin: bounds!.min[1] + 20, yMax: bounds!.max[1] - 20 }
      : undefined;
    const result = buildShell(sculpted, {
      zones,
      defaultThicknessMm: state.settings.defaultShellThicknessMm,
      trimLines: [], // 修剪线在体表绘制交互，当前版本用后开口与高度边界
      backOpening: bo,
      filletRadiusMm: fillet,
    });
    dispatch({ type: 'setMeshes', meshes: { ...state.meshes, shell: result.shell, shellInner: result.inner } });
    setIssues(result.issues);
    setWatertight(result.watertight);
    setSplit(splitSuggestion(result.shell, state.settings.printerMaxSize));
    dispatch({ type: 'toast', message: `壳体已生成，${result.issues.length} 个待处理项` });
  }

  const shell = state.meshes.shell;
  const thinVertices = useMemo(() => new Set(issues.filter((i) => i.kind === 'thin').map((i) => i.vertexId!)), [issues]);

  const shellWithHighlight: Mesh | undefined = useMemo(() => {
    if (!shell) return undefined;
    if (thinVertices.size === 0) return shell;
    const colors = new Float32Array((shell.positions.length / 3) * 3).fill(0.85);
    for (const v of thinVertices) {
      colors[v*3] = 1; colors[v*3+1] = 0.2; colors[v*3+2] = 0.1;
    }
    return { ...shell, colors };
  }, [shell, thinVertices]);

  return (
    <div>
      <h2 className="page-title">支具壳体</h2>
      <p className="page-sub">修型表面沿法线等距偏移成壳，厚度可分区（3–5mm），后开口与切口自动倒圆；自动高亮薄于 2mm 与自交区域。</p>
      <div className="split">
        <div>
          <div className="card">
            <h3>分区厚度（mm）</h3>
            {zones.map((z, i) => (
              <div key={i} className="grid-3" style={{ alignItems: 'end', marginBottom: 8 }}>
                <div className="form-row" style={{ margin: 0 }}><label>Y 起</label>
                  <input type="number" value={z.yMin} onChange={(e) => updateZone(i, { yMin: +e.target.value })} /></div>
                <div className="form-row" style={{ margin: 0 }}><label>Y 止</label>
                  <input type="number" value={z.yMax} onChange={(e) => updateZone(i, { yMax: +e.target.value })} /></div>
                <div className="form-row" style={{ margin: 0 }}><label>厚度</label>
                  <input type="number" min={3} max={5} step={0.5} value={z.thicknessMm}
                    onChange={(e) => updateZone(i, { thicknessMm: Math.min(5, Math.max(3, +e.target.value)) })} /></div>
              </div>
            ))}
            <div style={{ display: 'flex', gap: 8, marginTop: 8 }}>
              <button className="btn" onClick={addZone}>增加分区</button>
              <button className="btn danger" onClick={() => setZones((z) => z.slice(0, -1))} disabled={zones.length <= 1}>删除末区</button>
            </div>
          </div>

          <div className="card">
            <h3>开口与倒圆</h3>
            <label style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 10 }}>
              <input type="checkbox" checked={backOpen} onChange={(e) => setBackOpen(e.target.checked)} />
              启用后开口（穿脱用，自动倒圆）
            </label>
            <div className="form-row"><label>切口倒圆半径（mm）：{fillet}</label>
              <input type="range" min="0" max="4" step="0.5" value={fillet} onChange={(e) => setFillet(+e.target.value)} /></div>
            <label style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
              <input type="checkbox" checked={petg} onChange={(e) => setPetg(e.target.checked)} />
              半透明 PETG 预览（关闭则为不透明打印料）
            </label>
          </div>

          <button className="btn primary" style={{ width: '100%' }} onClick={generate} disabled={!sculpted}>
            生成等距偏移壳体
          </button>

          {watertight !== null && (
            <div className="card">
              <h3>制造校验</h3>
              <div className="kv"><span className="k">水密性</span>
                <span style={{ color: watertight ? 'var(--ok)' : 'var(--danger)' }}>{watertight ? '✓ 水密' : '✗ 存在未缝合边'}</span></div>
              <div className="kv"><span className="k">薄区（&lt;2mm）</span>
                <span style={{ color: thinVertices.size ? 'var(--danger)' : 'var(--ok)' }}>{thinVertices.size} 处</span></div>
              <div className="kv"><span className="k">自交嫌疑</span>
                <span>{issues.filter((i) => i.kind === 'self-intersection').length} 处</span></div>
              {split && (
                <div className="kv"><span className="k">打印机成型尺寸</span>
                  <span style={{ color: split.fits ? 'var(--ok)' : 'var(--warn)' }}>
                    {split.fits ? '✓ 一次成型' : `超限（${split.overAxis} 轴），建议在 Y=${split.cutHeights.map((h) => h.toFixed(0)).join(', ')}mm 分段`}
                  </span></div>
              )}
              <div className="panel-scroll" style={{ marginTop: 8 }}>
                {issues.slice(0, 12).map((iss, i) => (
                  <div key={i} style={{ fontSize: 12, padding: '4px 0', color: iss.kind === 'thin' ? 'var(--danger)' : 'var(--warn)' }}>
                    {iss.kind} · {iss.detail}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        <div className="card">
          <div className="toolbar">
            <strong>壳体预览</strong>
            <span className="muted">{meshSummary(shell)}</span>
          </div>
          {shellWithHighlight ? (
            <Viewport
              specs={[
                ...(sculpted ? [{ mesh: sculpted, id: 'sculpted', alpha: petg ? 0.18 : 0.1, baseColor: [0.7, 0.75, 0.85] as [number, number, number], visible: true }] : []),
                { mesh: shellWithHighlight, id: 'shell', alpha: petg ? 0.42 : 1, baseColor: [0.78, 0.86, 0.98], useVertexColor: thinVertices.size > 0, visible: true },
              ]}
            />
          ) : <p className="muted">设置厚度分区后点击「生成等距偏移壳体」。透过半透明壳可观察内部压垫区。</p>}
        </div>
      </div>
    </div>
  );

  function updateZone(i: number, patch: Partial<ThicknessZone>) {
    setZones((z) => z.map((x, idx) => (idx === i ? { ...x, ...patch } : x)));
  }
}
