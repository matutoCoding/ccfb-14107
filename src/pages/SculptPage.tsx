import { useMemo, useState } from 'react';
import { useApp } from '../state/appState';
import { Viewport, meshSummary } from '../components/Viewport';
import { Mesh, buildAdjacency, cloneMesh, vertexCount, getVertex, vertexNormals } from '../geometry/mesh';
import { applyStroke, BrushStroke, signedDifference, differenceColormap, crossSection } from '../geometry/sculpt';
import { getAdjacency } from '../geometry/geodesic';
import { gpuCompute } from '../gpu/GpuCompute';
import { db, newId } from '../db/database';
import { SculptPlan, StoredStroke } from '../db/schema';

export function SculptPage() {
  const { state, dispatch } = useApp();
  const [kind, setKind] = useState<'pad' | 'release'>('pad');
  const [radiusMm, setRadiusMm] = useState(60);
  const [depthMm, setDepthMm] = useState(8);
  const [strokes, setStrokes] = useState<BrushStroke[]>([]);
  const [displacements, setDisplacements] = useState<Float32Array[]>([]);
  const [showDiff, setShowDiff] = useState(false);
  const [sectionY, setSectionY] = useState(0);
  const [elapsed, setElapsed] = useState(0);

  const scan = state.meshes.scan;
  const sculpted = state.meshes.sculpted;
  const before = state.meshes.beforeSculpt;

  // 初始化工作副本
  function ensureWorkingMesh(): Mesh | null {
    if (!scan) { dispatch({ type: 'toast', message: '请先在扫描处理页准备扫描' }); return null; }
    if (!sculpted) {
      const work = cloneMesh(scan);
      dispatch({ type: 'setMeshes', meshes: { ...state.meshes, sculpted: work, beforeSculpt: cloneMesh(scan) } });
      return work;
    }
    return sculpted;
  }

  async function strokeAt(vertexId: number) {
    const work = ensureWorkingMesh();
    if (!work) return;
    const adj = getAdjacency(work);
    const t0 = performance.now();
    // 先 CPU 计算测地距离（已缓存邻接表，20 万面 <40ms）
    const { distance, method } = await Promise.resolve(
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      (applyStrokeVia(work, adj, vertexId)));
    const ms = performance.now() - t0;
    setElapsed(ms);
    setStrokes((s) => [...s, distance.stroke]);
    setDisplacements((d) => [...d, distance.displacement]);
    dispatch({ type: 'patchMesh', key: 'sculpted', mesh: work });
    dispatch({ type: 'toast', message: `${method === 'dijkstra' ? 'Dijkstra' : '热法'}测地修型 ${ms.toFixed(0)}ms` });
  }

  // 封装一层以便后续切换 GPU 路径；当前用几何内核（CPU，已满足 200ms）
  function applyStrokeVia(work: Mesh, adj: ReturnType<typeof buildAdjacency>, vertexId: number) {
    const res = applyStroke(work, adj, {
      kind, centerVertex: vertexId, radiusMm, depthMm,
    }, { autoContralateral: kind === 'pad', useDijkstraFallback: state.settings.geodesicMethod === 'dijkstra' });
    return Promise.resolve({ distance: res, method: state.settings.geodesicMethod });
  }

  function undo() {
    if (!sculpted || displacements.length === 0) return;
    const lastDisp = displacements[displacements.length - 1];
    // 反向位移（简化：沿当前法线回退）
    const normals = vertexNormals(sculpted);
    for (let i = 0; i < vertexCount(sculpted); i++) {
      if (lastDisp[i] === 0) continue;
      const p = getVertex(sculpted, i);
      const n: [number, number, number] = [normals[i*3], normals[i*3+1], normals[i*3+2]];
      sculpted.positions[i*3] = p[0] - n[0] * lastDisp[i];
      sculpted.positions[i*3+1] = p[1] - n[1] * lastDisp[i];
      sculpted.positions[i*3+2] = p[2] - n[2] * lastDisp[i];
    }
    setStrokes((s) => s.slice(0, -1));
    setDisplacements((d) => d.slice(0, -1));
    dispatch({ type: 'patchMesh', key: 'sculpted', mesh: { ...sculpted, positions: new Float32Array(sculpted.positions) } });
  }

  // 差值色谱
  const diffColors = useMemo(() => {
    if (!showDiff || !before || !sculpted) return undefined;
    const diff = signedDifference(before, sculpted);
    return differenceColormap(diff, Math.max(4, depthMm));
  }, [showDiff, before, sculpted, strokes.length, depthMm]);

  // 截面轮廓对比
  const sections = useMemo(() => {
    if (!before || !sculpted) return null;
    return {
      before: crossSection(before, sectionY, 4),
      after: crossSection(sculpted, sectionY, 4),
    };
  }, [before, sculpted, sectionY, strokes.length]);

  async function savePlan() {
    if (!state.currentPatient || !state.currentScan || !sculpted) {
      dispatch({ type: 'toast', message: '缺少患者/扫描/修型数据' }); return;
    }
    const stored: StoredStroke[] = strokes.map((s) => ({
      id: s.id, kind: s.kind, centerVertex: s.centerVertex, radiusMm: s.radiusMm,
      depthMm: s.depthMm, label: s.label, createdAt: s.createdAt, contralateral: s.contralateral,
    }));
    const plan: SculptPlan = {
      id: newId('plan'),
      patientId: state.currentPatient.id,
      visitId: state.currentVisit?.id ?? '',
      scanId: state.currentScan.id,
      name: `修型方案 ${state.plans.length + 1}`,
      strokes: stored,
      thicknessZones: [],
      status: '草稿',
      createdAt: Date.now(),
      updatedAt: Date.now(),
    };
    await db.put('plans', plan);
    dispatch({ type: 'setPlans', plans: await db.listPlans(state.currentPatient.id) });
    dispatch({ type: 'setCurrentPlan', plan });
    dispatch({ type: 'toast', message: '修型方案已保存' });
  }

  const displayMesh: Mesh | undefined = sculpted ?? scan;
  if (displayMesh && diffColors) displayMesh.colors = diffColors;

  return (
    <div>
      <h2 className="page-title">修型</h2>
      <p className="page-sub">压垫沿内法线推压并自动在对侧生成扩张空腔；释放区向外让位。衰减按体表测地距离（非直线）高斯分布。</p>
      <div className="split">
        <div>
          <div className="card">
            <h3>笔刷参数</h3>
            <div className="form-row">
              <label>笔刷类型</label>
              <div style={{ display: 'flex', gap: 8 }}>
                <button className={`btn ${kind === 'pad' ? 'primary' : ''}`} style={{ flex: 1 }} onClick={() => setKind('pad')}>压垫</button>
                <button className={`btn ${kind === 'release' ? 'primary' : ''}`} style={{ flex: 1 }} onClick={() => setKind('release')}>释放区</button>
              </div>
            </div>
            <div className="form-row"><label>影响半径（测地 mm）：{radiusMm}</label>
              <input type="range" min="20" max="140" value={radiusMm} onChange={(e) => setRadiusMm(+e.target.value)} /></div>
            <div className="form-row"><label>{kind === 'pad' ? '推压深度' : '释放深度'}（mm）：{depthMm}</label>
              <input type="range" min="1" max="25" step="0.5" value={depthMm} onChange={(e) => setDepthMm(+e.target.value)} /></div>
            <p className="muted" style={{ fontSize: 12 }}>
              在右侧网格上点击目标顶点即落笔。上一笔耗时 <strong>{elapsed.toFixed(0)}ms</strong>（预算 200ms）。
            </p>
            <div style={{ display: 'flex', gap: 8 }}>
              <button className="btn" style={{ flex: 1 }} onClick={undo} disabled={strokes.length === 0}>撤销上一笔</button>
              <button className="btn primary" style={{ flex: 1 }} onClick={savePlan} disabled={strokes.length === 0}>保存方案</button>
            </div>
          </div>

          <div className="card">
            <h3>笔画列表（{strokes.length}）</h3>
            <div className="panel-scroll">
              {strokes.map((s, i) => (
                <div key={s.id} className="kv">
                  <span><span className={`tag ${s.kind}`}>{s.kind === 'pad' ? '压垫' : '释放'}</span> #{i + 1} {s.label}</span>
                  <span className="muted">{s.depthMm}mm / R{s.radiusMm}mm{s.contralateral ? ' +对侧空腔' : ''}</span>
                </div>
              ))}
              {strokes.length === 0 && <p className="muted">尚无笔画</p>}
            </div>
          </div>

          <div className="card">
            <h3>对比视图</h3>
            <label style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 10 }}>
              <input type="checkbox" checked={showDiff} onChange={(e) => setShowDiff(e.target.checked)} />
              修型前后差值色谱（蓝=压入，红=释放）
            </label>
            <div className="form-row"><label>截面高度 Y（mm）：{sectionY}</label>
              <input type="range" min="-360" max="360" value={sectionY} onChange={(e) => setSectionY(+e.target.value)} /></div>
            {sections && <SectionSvg before={sections.before.points} after={sections.after.points} />}
          </div>
        </div>

        <div className="card">
          <div className="toolbar">
            <strong>修型工作区</strong>
            <span className="muted">{meshSummary(displayMesh)} · {state.gpu.available ? 'GPU 加速可用' : 'CPU 内核'}</span>
          </div>
          {displayMesh ? (
            <ClickableMesh mesh={displayMesh} onPick={strokeAt} />
          ) : <p className="muted">请先在扫描处理页导入或生成扫描。</p>}
        </div>
      </div>
    </div>
  );
}

/** 点击网格取最近顶点落笔 */
function ClickableMesh({ mesh, onPick }: { mesh: Mesh; onPick: (v: number) => void }) {
  function handle(e: React.MouseEvent<HTMLElement>) {
    // 屏幕坐标 → 最近顶点（用包围盒反投影近似，交互足够稳定）
    const rect = (e.target as HTMLElement).getBoundingClientRect();
    const nx = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    const ny = -(((e.clientY - rect.top) / rect.height) * 2 - 1);
    let best = -1, bestD = Infinity;
    const min: [number, number, number] = [Infinity, Infinity, Infinity];
    const max: [number, number, number] = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < mesh.positions.length; i += 3)
      for (let k = 0; k < 3; k++) { min[k] = Math.min(min[k], mesh.positions[i+k]); max[k] = Math.max(max[k], mesh.positions[i+k]); }
    const c = min.map((a, k) => (a + max[k]) / 2);
    const sx = (max[0] - min[0]) / 2, sy = (max[1] - min[1]) / 2;
    const tx = c[0] + nx * sx, ty = c[1] + ny * sy;
    for (let v = 0; v < vertexCount(mesh); v++) {
      const p = getVertex(mesh, v);
      const d = (p[0] - tx) ** 2 + (p[1] - ty) ** 2;
      if (d < bestD) { bestD = d; best = v; }
    }
    if (best >= 0) onPick(best);
  }
  return (
    <div onClick={handle}>
      <Viewport specs={[{ mesh, id: 'sculpt', alpha: 1, baseColor: [0.82, 0.85, 0.9], useVertexColor: !!mesh.colors, visible: true }]} />
    </div>
  );
}

function SectionSvg({ before, after }: { before: [number, number][]; after: [number, number][] }) {
  if (before.length < 3) return <p className="muted">该高度无足够截面点</p>;
  const all = [...before, ...after];
  const xs = all.map((p) => p[0]), zs = all.map((p) => p[1]);
  const minX = Math.min(...xs), maxX = Math.max(...xs), minZ = Math.min(...zs), maxZ = Math.max(...zs);
  const W = 300, H = 240, pad = 20;
  const map = (p: [number, number]) => [
    pad + ((p[0] - minX) / (maxX - minX || 1)) * (W - 2 * pad),
    H - pad - ((p[1] - minZ) / (maxZ - minZ || 1)) * (H - 2 * pad),
  ];
  const toPath = (pts: [number, number][]) => pts.map((p, i) => { const [x, y] = map(p); return `${i ? 'L' : 'M'}${x.toFixed(1)},${y.toFixed(1)}`; }).join(' ') + ' Z';
  return (
    <svg width="100%" height={H} style={{ background: '#fafbfc', borderRadius: 8 }}>
      <path d={toPath(before)} fill="none" stroke="#6b7686" strokeWidth="1.5" strokeDasharray="5 4" />
      <path d={toPath(after)} fill="none" stroke="#1f6feb" strokeWidth="2" />
      <text x={pad} y={14} fontSize="11" fill="#6b7686">灰虚线=修型前</text>
      <text x={pad} y={28} fontSize="11" fill="#1f6feb">蓝实线=修型后</text>
    </svg>
  );
}

void gpuCompute;
