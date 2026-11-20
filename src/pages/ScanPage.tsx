import { useRef, useState } from 'react';
import { useApp } from '../state/appState';
import { Viewport, meshSummary } from '../components/Viewport';
import { parseOBJ, parsePLY } from '../geometry/io';
import { Mesh, vertexCount, getVertex } from '../geometry/mesh';
import { cleanScan, alignmentTransform, applyTransform, nearestVertex, Landmark, ProcessingReport } from '../geometry/process';
import { generateTorso, presetScans } from '../geometry/generator';
import { db, newId } from '../db/database';
import { ScanRecord } from '../db/schema';

const LANDMARK_DEFS = [
  { id: 'asisL', label: '左髂前上棘' },
  { id: 'asisR', label: '右髂前上棘' },
  { id: 'axillaL', label: '左腋窝' },
  { id: 'axillaR', label: '右腋窝' },
];

export function ScanPage() {
  const { state, dispatch } = useApp();
  const fileRef = useRef<HTMLInputElement>(null);
  const [report, setReport] = useState<ProcessingReport | null>(null);
  const [landmarks, setLandmarks] = useState<Landmark[]>([]);
  const [selectedLm, setSelectedLm] = useState<string>('asisL');

  function requirePatient(): boolean {
    if (!state.currentPatient) { dispatch({ type: 'toast', message: '请先在患者档案页选择患者' }); return false; }
    return true;
  }

  async function handleFile(file: File) {
    if (!requirePatient()) return;
    try {
      let mesh: Mesh;
      if (file.name.toLowerCase().endsWith('.ply')) {
        mesh = parsePLY(await file.arrayBuffer(), file.name);
      } else {
        mesh = parseOBJ(await file.text(), file.name);
      }
      loadAndClean(mesh, file.name, file.name.toLowerCase().endsWith('.ply') ? 'ply' : 'obj', file);
    } catch (e) {
      dispatch({ type: 'toast', message: `解析失败：${(e as Error).message}` });
    }
  }

  function generatePreset(idx: number) {
    if (!requirePatient()) return;
    const preset = presetScans()[idx];
    const mesh = generateTorso(preset.options);
    loadAndClean(mesh, preset.name, 'synthetic');
  }

  function loadAndClean(raw: Mesh, name: string, source: ScanRecord['source'], original?: Blob) {
    const { mesh, report: r } = cleanScan(raw);
    setReport(r);
    // 自动放置 4 个解剖标记点（按包围盒比例的启发式位置，吸附最近顶点）
    const auto = autoLandmarks(mesh);
    setLandmarks(auto);
    dispatch({
      type: 'setMeshes',
      meshes: { ...state.meshes, scan: mesh, sculpted: undefined, beforeSculpt: undefined, shell: undefined, shellInner: undefined },
    });
    dispatch({ type: 'toast', message: `${name}：清理 ${r.removedComponents} 个杂点域，补 ${r.holesFilled} 个洞` });
    // 暂存原始文件到模块变量，保存时落库
    pendingOriginal = original;
    pendingName = name;
    pendingSource = source;
  }

  async function alignAndSave() {
    if (!state.meshes.scan || !state.currentPatient) return;
    try {
      const t = alignmentTransform(landmarks);
      const mesh = state.meshes.scan;
      applyTransform(mesh, t);
      // 标记点随变换更新
      const moved = landmarks.map((lm) => ({ ...lm, position: transformPoint(lm.position, t) }));
      setLandmarks(moved);
      const scanId = newId('scan');
      const record: ScanRecord = {
        id: scanId,
        patientId: state.currentPatient.id,
        visitId: state.currentVisit?.id ?? '',
        name: pendingName ?? '躯干扫描',
        source: pendingSource,
        vertexCount: vertexCount(mesh),
        faceCount: mesh.indices.length / 3,
        processingReport: report ?? undefined,
        landmarks: moved.map((l) => ({ id: l.id, label: l.label, position: l.position, vertexId: l.vertexId })),
        createdAt: Date.now(),
      };
      await db.put('scans', record);
      await db.saveMeshBlob({
        id: scanId,
        positions: mesh.positions.buffer as ArrayBuffer,
        indices: mesh.indices.buffer as ArrayBuffer,
        original: pendingOriginal,
      });
      dispatch({ type: 'setScans', scans: await db.listScans(state.currentPatient.id) });
      dispatch({ type: 'setCurrentScan', scan: record });
      dispatch({ type: 'toast', message: '扫描已摆正并保存到本地 IndexedDB' });
    } catch (e) {
      dispatch({ type: 'toast', message: (e as Error).message });
    }
  }

  /** 在画布点击处把当前选中的标记点拖到最近体表顶点 */
  function onCanvasClick(e: React.MouseEvent<HTMLElement>) {
    if (!state.meshes.scan || landmarks.length === 0) return;
    // 简化交互：沿视角在点击射线与包围盒求交，再吸附最近顶点
    const mesh = state.meshes.scan;
    const rect = (e.target as HTMLElement).getBoundingClientRect();
    const nx = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    const ny = -(((e.clientY - rect.top) / rect.height) * 2 - 1);
    const target = rayApproxPoint(mesh, nx, ny);
    const { id } = nearestVertex(mesh, target);
    setLandmarks((prev) => prev.map((lm) =>
      lm.id === selectedLm ? { ...lm, vertexId: id, position: getVertex(mesh, id) } : lm));
  }

  const scanMesh = state.meshes.scan;
  const markerMesh: Mesh | undefined = scanMesh && landmarks.length ? buildMarkerMesh(landmarks) : undefined;

  return (
    <div>
      <h2 className="page-title">扫描处理</h2>
      <p className="page-sub">导入 OBJ/PLY，自动去除手臂与扶架杂点、补洞，按髂前上棘与腋窝标记点摆正坐标系；标记点可手动拖动。</p>
      <div className="split">
        <div>
          <div className="card">
            <h3>数据来源</h3>
            <input ref={fileRef} type="file" accept=".obj,.ply" style={{ display: 'none' }}
              onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])} />
            <button className="btn primary" style={{ width: '100%', marginBottom: 10 }} onClick={() => fileRef.current?.click()}>
              导入 OBJ / PLY 扫描
            </button>
            <p className="muted" style={{ fontSize: 12 }}>或用程序生成测试躯干：</p>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
              {presetScans().map((p, i) => (
                <button key={p.name} className="btn" onClick={() => generatePreset(i)}>{p.name}</button>
              ))}
            </div>
          </div>

          {report && (
            <div className="card">
              <h3>处理报告</h3>
              <div className="kv"><span className="k">清除杂点连通域</span><span>{report.removedComponents} 个</span></div>
              <div className="kv"><span className="k">删除顶点 / 面</span><span>{report.removedVertices} / {report.removedFaces}</span></div>
              <div className="kv"><span className="k">填补破洞</span><span>{report.holesFilled} 个</span></div>
              <div className="kv"><span className="k">新增三角面</span><span>{report.trianglesAdded}</span></div>
            </div>
          )}

          {landmarks.length > 0 && (
            <div className="card">
              <h3>解剖标记点（可拖动）</h3>
              <p className="muted" style={{ fontSize: 12 }}>选中一个标记点后，在右侧网格上点击目标位置即可吸附。</p>
              {LANDMARK_DEFS.map((d) => {
                const lm = landmarks.find((l) => l.id === d.id);
                return (
                  <div key={d.id} style={{ marginBottom: 6 }}>
                    <label style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer' }}>
                      <input type="radio" name="lm" checked={selectedLm === d.id} onChange={() => setSelectedLm(d.id)} />
                      {d.label}
                      <span className="muted" style={{ fontSize: 12 }}>
                        #{lm?.vertexId} ({lm?.position.map((x) => x.toFixed(1)).join(', ')})
                      </span>
                    </label>
                  </div>
                );
              })}
              <button className="btn primary" style={{ width: '100%', marginTop: 8 }} onClick={alignAndSave}>
                摆正坐标系并保存扫描
              </button>
            </div>
          )}

          {state.scans.length > 0 && (
            <div className="card">
              <h3>本患者已存扫描</h3>
              {state.scans.map((s) => (
                <div key={s.id} className="kv clickable"
                  style={{ cursor: 'pointer', background: state.currentScan?.id === s.id ? 'var(--primary-dim)' : undefined }}
                  onClick={() => dispatch({ type: 'setCurrentScan', scan: s })}>
                  <span className="k">{s.name}</span>
                  <span>{s.vertexCount.toLocaleString()} 点</span>
                </div>
              ))}
            </div>
          )}
        </div>

        <div>
          <div className="card">
            <div className="toolbar">
              <strong>三维视图</strong>
              <span className="muted">{meshSummary(scanMesh)}</span>
            </div>
            <div onClick={onCanvasClick}>
              <Viewport
                specs={[
                  ...(scanMesh ? [{ mesh: scanMesh, id: 'scan', alpha: 1, baseColor: [0.82, 0.85, 0.9] as [number, number, number], visible: true }] : []),
                  ...(markerMesh ? [{ mesh: markerMesh, id: 'markers', alpha: 1, baseColor: [1, 0.2, 0.2] as [number, number, number], visible: true }] : []),
                ]}
              />
            </div>
            {!scanMesh && <p className="muted">导入或生成扫描后在此预览。</p>}
          </div>
        </div>
      </div>
    </div>
  );
}

let pendingOriginal: Blob | undefined;
let pendingName = '';
let pendingSource: ScanRecord['source'] = 'obj';

function autoLandmarks(mesh: Mesh): Landmark[] {
  // 启发式：在包围盒的髂/腋高度、左右两侧取最近顶点
  const out: Landmark[] = [];
  for (const def of LANDMARK_DEFS) {
    const approx = heuristicPoint(mesh, def.id);
    const { id } = nearestVertex(mesh, approx);
    out.push({ id: def.id, label: def.label, vertexId: id, position: getVertex(mesh, id) });
  }
  return out;
}

function heuristicPoint(mesh: Mesh, id: string): [number, number, number] {
  const min: [number, number, number] = [Infinity, Infinity, Infinity];
  const max: [number, number, number] = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < mesh.positions.length; i += 3)
    for (let k = 0; k < 3; k++) { min[k] = Math.min(min[k], mesh.positions[i + k]); max[k] = Math.max(max[k], mesh.positions[i + k]); }
  const yAsis = min[1] + (max[1] - min[1]) * 0.18;
  const yAxilla = min[1] + (max[1] - min[1]) * 0.72;
  const side = id.startsWith('asisL') || id.startsWith('axillaL') ? max[0] : min[0];
  const y = id.includes('asis') ? yAsis : yAxilla;
  return [side, y, (min[2] + max[2]) / 2];
}

function buildMarkerMesh(landmarks: Landmark[]): Mesh {
  // 每个标记点做成小八面体（6 个面），独立于主网格着色
  const positions: number[] = [];
  const indices: number[] = [];
  const r = 8;
  for (const lm of landmarks) {
    const [cx, cy, cz] = lm.position;
    const base = positions.length / 3;
    positions.push(
      cx + r, cy, cz, cx - r, cy, cz, cx, cy + r, cz, cx, cy - r, cz, cx, cy, cz + r, cx, cy, cz - r,
    );
    const faces = [[0,2,4],[0,4,3],[0,3,5],[0,5,2],[1,4,2],[1,3,4],[1,5,3],[1,2,5]];
    for (const f of faces) indices.push(base + f[0], base + f[1], base + f[2]);
  }
  return { positions: new Float32Array(positions), indices: new Uint32Array(indices) };
}

function transformPoint(p: [number, number, number], m: Float32Array): [number, number, number] {
  const x = m[0]*p[0] + m[4]*p[1] + m[8]*p[2] + m[12];
  const y = m[1]*p[0] + m[5]*p[1] + m[9]*p[2] + m[13];
  const z = m[2]*p[0] + m[6]*p[1] + m[10]*p[2] + m[14];
  return [x, y, z];
}

function rayApproxPoint(mesh: Mesh, nx: number, ny: number): [number, number, number] {
  // 取包围盒中心深度平面上的近似点
  const min: [number, number, number] = [Infinity, Infinity, Infinity];
  const max: [number, number, number] = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < mesh.positions.length; i += 3)
    for (let k = 0; k < 3; k++) { min[k] = Math.min(min[k], mesh.positions[i + k]); max[k] = Math.max(max[k], mesh.positions[i + k]); }
  const c = min.map((a, k) => (a + max[k]) / 2) as [number, number, number];
  const sx = (max[0] - min[0]) / 2;
  const sy = (max[1] - min[1]) / 2;
  return [c[0] + nx * sx, c[1] + ny * sy, c[2]];
}
