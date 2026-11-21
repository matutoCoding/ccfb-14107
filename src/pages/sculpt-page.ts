// 修型页：压垫/释放区测地笔刷、对侧自动空腔、叠网差值色谱、逐层截面、Cobb 实时预测
import { h, toast, svgEl } from '../ui/dom';
import { createViewport } from '../ui/viewport';
import { gpuSession, setSculptedMesh } from '../state/gpu-session';
import { store } from '../state/store';
import { workbench } from '../state/workbench';
import { TriangleMesh } from '../core/mesh';
import { SculptEngine } from '../core/sculpt';
import { defaultDab, extractCrossSections, SculptDab } from '../core/sculpt-types';
import { predictSpine, curveToPolyline } from '../core/spine-predict';
import { HeatGeodesic } from '../gpu/geodesic';
import { SculptPlanRecord } from '../core/types';

export function sculptPage(): HTMLElement {
  const session = gpuSession();
  const baseMesh = workbench.activeMesh();
  if (!baseMesh) return h('div', { class: 'empty' }, ['请先在「扫描处理」中导入并保存扫描。']);
  const base: TriangleMesh = baseMesh;

  const wrap = h('div', { class: 'workspace wide-left' });
  const view = createViewport(620);
  view.renderer.clearMeshes();
  const side = h('div', { class: 'panel-side' });
  const engine = session.sculpt ?? new SculptEngine(base, session.geodesic as HeatGeodesic);
  const dabs: SculptDab[] = [];
  let sculptedPositions: Float32Array = new Float32Array(base.positions) as Float32Array;
  let diffMode = false;
  let showSections = false;

  view.renderer.removeMesh('work');
  view.renderer.addMesh('base', base, { baseColor: [0.7, 0.72, 0.75], opacity: 0.35, aoStrength: 0.8 });
  view.renderer.addMesh('sculpt', new TriangleMesh(sculptedPositions, new Uint32Array(base.indices)), {
    baseColor: [0.85, 0.78, 0.7],
    opacity: 1,
    aoStrength: 0.8,
  });
  view.renderer.setPickMesh(base);
  view.renderer.orbit.target = base.bbox().center;

  // —— 笔刷参数 ——
  const kindState: { kind: 'pressure' | 'relief' } = { kind: 'pressure' };
  const sigmaInput = numInput(38);
  const depthInput = numInput(12);
  const contraDepth = numInput(8);
  const contraToggle = h('input', { type: 'checkbox', checked: true });

  let pickedVertex: number | null = null;
  view.canvas.addEventListener('click', (e) => {
    const hit = view.renderer.pick(e.clientX, e.clientY);
    if (hit) {
      pickedVertex = hit.vertex;
      pickInfo.textContent = `落笔顶点 #${hit.vertex}  坐标 ${hit.point.map((v) => v.toFixed(1)).join(', ')}`;
    }
  });

  const pickInfo = h('div', { class: 'muted' }, ['在视口点击网格选择落笔顶点']);

  side.append(
    h('div', { class: 'card' }, [
      h('h3', {}, ['笔刷']),
      h('div', { class: 'pill-group', style: 'margin-bottom:12px' }, [
        pill('压垫（内推）', true, () => { kindState.kind = 'pressure'; syncPill(); }),
        pill('释放区（外扩）', false, () => { kindState.kind = 'relief'; syncPill(); }),
      ]),
      field('测地衰减 σ（mm）— 沿体表距离，非直线', sigmaInput),
      field('推压深度（mm，正数）', depthInput),
      h('div', { class: 'field' }, [
        h('label', {}, ['对侧自动扩张空腔（压垫）']),
        h('div', { class: 'row' }, [contraToggle, h('span', { class: 'muted' }, ['启用'])]),
      ]),
      field('空腔深度 mm', contraDepth),
      pickInfo,
      h('button', {
        class: 'primary',
        style: 'width:100%;margin-top:8px',
        onClick: async () => {
          if (pickedVertex === null) return toast('请先在网格上点选落笔顶点', 'error');
          const dab = defaultDab(pickedVertex, kindState.kind);
          dab.sigmaMm = Number(sigmaInput.value);
          dab.depthMm = Number(depthInput.value);
          if (dab.contralateral) {
            dab.contralateral.enabled = contraToggle.checked;
            dab.contralateral.depthMm = Number(contraDepth.value);
          }
          dabs.push(dab);
          await rerunSculpt();
          renderDabList();
          updatePrediction();
        },
      }, ['落笔应用']),
    ]),
    h('div', { class: 'card' }, [
      h('h3', {}, ['已应用笔刷']),
      h('div', { id: 'dab-list' }, [h('div', { class: 'muted' }, ['尚无笔刷'])]),
    ]),
    h('div', { class: 'card' }, [
      h('h3', {}, ['可视化']),
      h('div', { class: 'grid', style: 'gap:8px' }, [
        h('button', {
          onClick: async () => {
            diffMode = !diffMode;
            await toggleDiff();
          },
        }, ['切换：实体 / 差值色谱']),
        h('button', { onClick: () => { showSections = !showSections; toggleSections(); } }, ['切换：逐层截面对比']),
        h('button', {
          onClick: () => {
            const baseRm = view.renderer.getMesh('base');
            if (baseRm) view.renderer.setMaterial('base', { opacity: baseRm.material.opacity < 0.2 ? 0.35 : 0.08 });
          },
        }, ['调整原始网格透明度']),
      ]),
      h('div', { class: 'legend', style: 'margin-top:10px' }, [
        h('span', {}, ['释放/外扩']), h('div', { class: 'bar' }), h('span', {}, ['压垫/内推']),
      ]),
      h('div', { id: 'perf-line', class: 'muted', style: 'margin-top:8px' }),
    ]),
    h('div', { class: 'card' }, [
      h('h3', {}, ['修型方案']),
      h('div', { class: 'grid', style: 'gap:8px' }, [
        h('button', {
          class: 'primary',
          onClick: async () => {
            const patient = store.state.currentPatient;
            const scan = store.state.currentScan;
            if (!patient) return toast('缺少患者上下文', 'error');
            const rec: SculptPlanRecord = {
              id: store.newId('plan'),
              patientId: patient.id,
              visitIndex: scan?.visitIndex ?? Math.max(0, ...patient.visits.map((v) => v.visitIndex)) + 1,
              scanId: scan?.id ?? 'workbench',
              name: `方案 ${new Date().toLocaleString('zh-CN')}`,
              createdAt: Date.now(),
              dabs: [...dabs],
              sculptedPositions,
              basePositions: new Float32Array(base.positions),
            };
            await store.savePlan(rec);
            toast('修型方案已保存，可进入支具壳体页', 'success');
          },
        }, ['保存方案']),
        h('button', {
          onClick: async () => {
            dabs.length = 0;
            sculptedPositions = new Float32Array(base.positions);
            view.renderer.updatePositions('sculpt', sculptedPositions, base.vertexNormals());
            setSculptedMesh(null);
            renderDabList();
            updatePrediction();
            toast('已清空全部笔刷', 'info');
          },
        }, ['清空重来']),
      ]),
    ]),
  );

  const predictionCard = h('div', { class: 'card' }, [h('h3', {}, ['矫正效果预测（随笔刷实时更新）']), h('div', { id: 'prediction' })]);
  side.append(predictionCard);

  wrap.append(view.el, side);

  function syncPill() {
    side.querySelectorAll('.pill-group button').forEach((b, i) => {
      b.classList.toggle('on', (i === 0) === (kindState.kind === 'pressure'));
    });
  }
  syncPill();

  async function rerunSculpt() {
    const t0 = performance.now();
    const result = await engine.applyPlan(new Float32Array(base.positions), dabs);
    sculptedPositions = result.positions;
    const sculptedMesh = new TriangleMesh(new Float32Array(sculptedPositions), new Uint32Array(base.indices));
    setSculptedMesh(sculptedMesh);
    view.renderer.updatePositions('sculpt', sculptedPositions, sculptedMesh.vertexNormals());
    const ms = performance.now() - t0;
    const target = 200;
    document.getElementById('perf-line')!.textContent =
      `本次修型 ${ms.toFixed(0)}ms（${base.faceCount} 面，预算 ${target}ms）${ms <= target ? ' ✓ 达标' : ' ✗ 超时，可降低 CG 迭代数'}`;
    if (diffMode) await toggleDiff();
    if (showSections) toggleSections();
  }

  async function toggleDiff() {
    if (diffMode) {
      const { scalar } = engine.diffField(base.positions, sculptedPositions);
      const colors = scalarToColors(scalar);
      view.renderer.updateColors('sculpt', colors);
      view.renderer.setMaterial('sculpt', { mode: 1 });
      view.renderer.setMaterial('base', { opacity: 0.12 });
    } else {
      view.renderer.setMaterial('sculpt', { mode: 0 });
      view.renderer.setMaterial('base', { opacity: 0.35 });
    }
  }

  function toggleSections() {
    const existing = document.getElementById('section-overlay');
    if (existing) { existing.remove(); return; }
    const bb = base.bbox();
    const levels = [0.2, 0.4, 0.6, 0.8].map((t) => bb.min[1] + (bb.max[1] - bb.min[1]) * t);
    const sections = extractCrossSections(base, sculptedPositions, levels, 5);
    const overlay = h('div', {
      id: 'section-overlay',
      style: 'position:absolute;left:12px;top:12px;width:280px;max-height:46%;overflow:auto;background:rgba(15,20,25,.9);border:1px solid var(--border);border-radius:8px;padding:10px;',
    }, [h('strong', { style: 'font-size:12px' }, ['逐层截面（灰=原始，红=修型后）'])]);
    for (const sec of sections) overlay.append(sectionSvg(sec));
    view.el.append(overlay);
  }

  function renderDabList() {
    const el = document.getElementById('dab-list');
    if (!el) return;
    if (!dabs.length) {
      el.replaceChildren(h('div', { class: 'muted' }, ['尚无笔刷']));
      return;
    }
    el.replaceChildren(...dabs.map((d, i) => h('div', { class: 'dab-item' }, [
      h('div', { class: 'head' }, [
        h('strong', {}, [`#${i + 1} ${d.kind === 'pressure' ? '压垫' : '释放区'}`]),
        h('button', {
          class: 'ghost danger',
          onClick: async () => { dabs.splice(i, 1); await rerunSculpt(); renderDabList(); updatePrediction(); },
        }, ['删除']),
      ]),
      h('div', { class: 'muted' }, [
        `顶点 #${d.centerVertex} · σ ${d.sigmaMm}mm · 深度 ${d.depthMm}mm`,
        d.contralateral?.enabled ? ` · 对侧空腔 ${d.contralateral.depthMm}mm` : '',
      ]),
    ])));
  }

  function updatePrediction() {
    const el = document.getElementById('prediction');
    if (!el) return;
    const originalCobb = store.state.currentPatient?.visits.at(-1)?.cobbAngleDeg ?? 30;
    const pred = predictSpine(base, dabs, originalCobb);
    const w = 260, hgt = 190;
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('viewBox', `0 0 ${w} ${hgt}`);
    svg.setAttribute('class', 'chart');
    const bb = base.bbox();
    const d1 = curveToPolyline(pred.originalCurve, w, hgt, bb);
    const d2 = curveToPolyline(pred.correctedCurve, w, hgt, bb);
    svg.append(svgEl('path', { d: d1, fill: 'none', stroke: '#8b98a5', 'stroke-width': 2 }));
    svg.append(svgEl('path', { d: d2, fill: 'none', stroke: '#2bd4a8', 'stroke-width': 2.4 }));
    el.replaceChildren(
      svg,
      h('div', { class: 'kv', style: 'margin-top:8px' }, [
        h('span', { class: 'k' }, ['原始 Cobb']), h('span', {}, [`${pred.originalCobbDeg.toFixed(1)}°`]),
        h('span', { class: 'k' }, ['预估 Cobb']), h('span', { style: 'color:var(--accent-2);font-weight:700' }, [`${pred.predictedCobbDeg.toFixed(1)}°`]),
        h('span', { class: 'k' }, ['预估矫正率']), h('span', {}, [`${(pred.correctionRate * 100).toFixed(1)}%`]),
      ]),
      h('div', { class: 'muted', style: 'margin-top:6px' }, ['灰线=原始中线，绿线=压垫作用后预估中线；改动笔刷即时重算。']),
    );
  }
  updatePrediction();

  return wrap;
}

// ——  helpers ——
function numInput(value: number) {
  return h('input', { type: 'number', value: String(value), step: '0.5', min: '0' });
}
function field(label: string, control: HTMLElement) {
  return h('div', { class: 'field' }, [h('label', {}, [label]), control]);
}
function pill(label: string, active: boolean, onClick: () => void) {
  return h('button', { class: active ? 'on' : '', onClick }, [label]);
}

function scalarToColors(scalar: Float32Array): Float32Array {
  let maxAbs = 1e-6;
  for (const v of scalar) maxAbs = Math.max(maxAbs, Math.abs(v));
  const colors = new Float32Array(scalar.length * 3);
  for (let i = 0; i < scalar.length; i++) {
    const t = scalar[i] / maxAbs; // 正=内推红，负=外扩蓝
    if (t >= 0) {
      colors[i * 3] = 0.95 - 0.05 * (1 - t);
      colors[i * 3 + 1] = 0.95 * (1 - t) + 0.25 * t;
      colors[i * 3 + 2] = 0.95 * (1 - t) + 0.15 * t;
    } else {
      const a = -t;
      colors[i * 3] = 0.95 * (1 - a) + 0.1 * a;
      colors[i * 3 + 1] = 0.95 * (1 - a) + 0.35 * a;
      colors[i * 3 + 2] = 0.95 - 0.05 * (1 - a);
    }
  }
  return colors;
}

function sectionSvg(sec: { y: number; before: [number, number, number][]; after: [number, number, number][] }) {
  const size = 120;
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('viewBox', `0 0 ${size} ${size}`);
  svg.setAttribute('style', 'width:100%;height:120px;margin-top:6px');
  const all = [...sec.before, ...sec.after];
  if (all.length < 3) return h('div', { class: 'muted' }, [`Y=${sec.y.toFixed(0)} 截面数据不足`]);
  const xs = all.map((p) => p[0]), zs = all.map((p) => p[2]);
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minZ = Math.min(...zs), maxZ = Math.max(...zs);
  const sx = (x: number) => 8 + ((x - minX) / Math.max(1e-6, maxX - minX)) * (size - 16);
  const sy = (z: number) => size - 8 - ((z - minZ) / Math.max(1e-6, maxZ - minZ)) * (size - 16);
  const path = (pts: [number, number, number][]) =>
    pts.map((p, i) => `${i ? 'L' : 'M'}${sx(p[0]).toFixed(1)},${sy(p[2]).toFixed(1)}`).join(' ') + ' Z';
  if (sec.before.length > 2) svg.append(svgEl('path', { d: path(sec.before), fill: 'none', stroke: '#8b98a5', 'stroke-width': 1.4 }));
  if (sec.after.length > 2) svg.append(svgEl('path', { d: path(sec.after), fill: 'none', stroke: '#f25555', 'stroke-width': 1.4 }));
  return h('div', {}, [h('div', { class: 'muted', style: 'font-size:11px' }, [`Y = ${sec.y.toFixed(0)} mm`]), svg]);
}
