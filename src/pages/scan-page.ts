// 扫描处理页：OBJ/PLY 导入、程序化生成、去杂点、补洞、标记点摆正/拖拽、入库
import { h, toast } from '../ui/dom';
import { createViewport } from '../ui/viewport';
import { workbench } from '../state/workbench';
import { store } from '../state/store';
import { setBaseMesh, gpuSession } from '../state/gpu-session';
import { parseOBJ, parsePLY } from '../core/io-formats';
import {
  alignMesh,
  buildAlignmentFrame,
  fillHoles,
  guessLandmarks,
  removeSmallComponents,
  snapToVertex,
  AnatomicalLandmarks,
} from '../core/processing';
import { generateTorso, builtinTestTorsos } from '../core/synthetic-torso';
import { TriangleMesh } from '../core/mesh';
import { ScanRecord } from '../core/types';

export function scanPage(): HTMLElement {
  const patient = store.state.currentPatient;
  const wrap = h('div', { class: 'workspace' });
  const view = createViewport(600);
  view.renderer.clearMeshes();
  const side = h('div', { class: 'panel-side' });

  if (!patient) {
    view.stop();
    return h('div', { class: 'empty' }, ['请先在「患者档案」中选择或新建患者，再进入扫描处理。']);
  }

  // —— 工具条 ——
  view.tools.append(
    toolButton('显示原始', () => showMesh(workbench.raw!)),
    toolButton('显示清洗后', () => showMesh(workbench.cleaned ?? workbench.raw!)),
    toolButton('显示摆正后', () => showMesh(workbench.aligned ?? workbench.cleaned ?? workbench.raw!)),
  );

  // —— 侧栏 ——
  side.append(
    h('div', { class: 'card' }, [
      h('h3', {}, ['导入扫描']),
      h('input', {
        type: 'file',
        accept: '.obj,.ply',
        onChange: async (e: Event) => {
          const input = e.target as HTMLInputElement;
          const file = input.files?.[0];
          if (!file) return;
          try {
            const buf = await file.arrayBuffer();
            const mesh = file.name.endsWith('.ply')
              ? parsePLY(buf)
              : parseOBJ(new TextDecoder().decode(buf));
            workbench.reset();
            workbench.name = file.name;
            workbench.raw = mesh;
            workbench.cleaned = mesh;
            toast(`已加载 ${file.name}：${mesh.vertexCount} 顶点 / ${mesh.faceCount} 面`, 'success');
            showMesh(mesh);
            refreshStatus();
          } catch (err) {
            toast(`导入失败：${(err as Error).message}`, 'error');
          }
        },
      }),
      h('div', { class: 'muted', style: 'margin-top:8px' }, ['支持 OBJ（ASCII）与 PLY（ASCII / 小端二进制）']),
    ]),
    h('div', { class: 'card' }, [
      h('h3', {}, ['程序化测试躯干']),
      h('div', { class: 'grid', style: 'gap:8px' }, builtinTestTorsos().map((t) =>
        h('button', {
          onClick: () => {
            const g = generateTorso(t.options);
            workbench.reset();
            workbench.name = t.name;
            workbench.syntheticFlag = true;
            workbench.raw = g.mesh;
            workbench.cleaned = g.mesh;
            toast(`已生成「${t.name}」：${g.mesh.vertexCount} 顶点，${g.holes.length} 个破洞`, 'success');
            showMesh(g.mesh);
            refreshStatus();
          },
        }, [t.name]),
      )),
    ]),
    h('div', { class: 'card' }, [
      h('h3', {}, ['自动处理']),
      h('div', { class: 'grid', style: 'gap:8px' }, [
        h('button', {
          onClick: () => {
            if (!workbench.raw) return toast('请先导入或生成扫描', 'error');
            const t0 = performance.now();
            const r1 = removeSmallComponents(workbench.raw);
            const r2 = fillHoles(r1.mesh);
            workbench.cleaned = r2.mesh;
            workbench.stats = {
              ...r1.stats,
              filledHoles: r2.filledHoles,
              filledTriangles: r2.filledTriangles,
            };
            toast(`去杂点移除 ${r1.stats.removedComponents} 个连通域，补洞 ${r2.filledHoles} 个，耗时 ${(performance.now() - t0).toFixed(0)}ms`, 'success');
            showMesh(r2.mesh);
            refreshStatus();
          },
        }, ['一键去杂点 + 补洞']),
        h('button', {
          onClick: () => {
            const mesh = workbench.cleaned ?? workbench.raw;
            if (!mesh) return toast('请先导入扫描', 'error');
            const lm = guessLandmarks(mesh);
            workbench.landmarksValue = lm;
            toast('已自动标记 ASIS 与腋窝，可在视口中拖动微调', 'success');
            renderLandmarkEditor(lm);
          },
        }, ['自动识别解剖标记点']),
        h('button', {
          onClick: () => {
            const mesh = workbench.cleaned ?? workbench.raw;
            const lm = workbench.landmarksValue;
            if (!mesh || !lm) return toast('请先识别标记点', 'error');
            const aligned = alignMesh(mesh, lm);
            workbench.aligned = aligned;
            const { origin } = buildAlignmentFrame(lm);
            toast(`坐标系已摆正，原点 ASIS 中点 (${origin.map((v) => v.toFixed(1)).join(', ')})`, 'success');
            showMesh(aligned);
            refreshStatus();
          },
        }, ['按标记点摆正坐标系']),
      ]),
    ]),
    h('div', { class: 'card' }, [
      h('h3', {}, ['标记点微调']),
      h('div', { id: 'lm-editor', class: 'muted' }, ['识别标记点后，可点击视口顶点再点下方按钮替换，或直接编辑坐标。']),
    ]),
    h('div', { class: 'card' }, [
      h('h3', {}, ['入库']),
      h('button', {
        class: 'primary',
        style: 'width:100%',
        onClick: async () => {
          const mesh = workbench.activeMesh();
          if (!mesh) return toast('没有可保存的网格', 'error');
          if (!workbench.landmarksValue) return toast('请先识别并摆正标记点', 'error');
          const visitIndex = Math.max(0, ...patient.visits.map((v) => v.visitIndex)) + 1;
          const rec: ScanRecord = {
            id: store.newId('scan'),
            patientId: patient.id,
            visitIndex,
            filename: workbench.name || '未命名扫描',
            createdAt: Date.now(),
            positions: mesh.positions,
            indices: mesh.indices,
            landmarks: {
              asiLeft: workbench.landmarksValue.asiLeft,
              asiRight: workbench.landmarksValue.asiRight,
              axillaLeft: workbench.landmarksValue.axillaLeft,
              axillaRight: workbench.landmarksValue.axillaRight,
            },
            cleaningStats: workbench.stats ?? undefined,
            synthetic: workbench.syntheticFlag,
          };
          await store.saveScan(rec);
          setBaseMesh(mesh);
          toast('扫描已存入 IndexedDB 并载入修型工作台', 'success');
        },
      }, ['保存扫描并载入修型']),
      h('div', { style: 'height:10px' }),
      existingScanList(),
    ]),
  );

  const status = h('div', { class: 'card' }, [h('h3', {}, ['网格状态']), h('div', { id: 'mesh-status' }, ['尚未加载'])]);
  side.prepend(status);

  wrap.append(view.el, side);

  function refreshStatus() {
    const el = document.getElementById('mesh-status');
    if (!el) return;
    const m = workbench.activeMesh();
    el.replaceChildren(h('div', { class: 'kv' }, [
      h('span', { class: 'k' }, ['文件']), h('span', {}, [workbench.name || '—']),
      h('span', { class: 'k' }, ['顶点']), h('span', {}, [m ? String(m.vertexCount) : '—']),
      h('span', { class: 'k' }, ['三角面']), h('span', {}, [m ? String(m.faceCount) : '—']),
      h('span', { class: 'k' }, ['水密']), h('span', {}, [m ? (m.isWatertight() ? '是' : '否（存在边界）') : '—']),
      h('span', { class: 'k' }, ['处理']), h('span', {}, [workbench.stats ? `去杂 ${workbench.stats.removedComponents} 域 / 补洞 ${workbench.stats.filledHoles}` : '未处理']),
    ]));
  }

  function showMesh(mesh: TriangleMesh) {
    const renderer = view.renderer;
    renderer.removeMesh('work');
    renderer.addMesh('work', mesh, { baseColor: [0.82, 0.76, 0.7], opacity: 1, aoStrength: 0.8 });
    renderer.setPickMesh(mesh);
    const bb = mesh.bbox();
    renderer.orbit.target = bb.center;
  }

  let pickedVertex: number | null = null;
  view.canvas.addEventListener('click', (e) => {
    const hit = view.renderer.pick(e.clientX, e.clientY);
    if (hit) {
      pickedVertex = hit.vertex;
      toast(`已选顶点 #${hit.vertex}`, 'info', 1500);
    }
  });

  function renderLandmarkEditor(lm: AnatomicalLandmarks) {
    const editor = document.getElementById('lm-editor');
    if (!editor) return;
    const keys: Array<{ key: 'asiLeft' | 'asiRight' | 'axillaLeft' | 'axillaRight'; label: string }> = [
      { key: 'asiLeft', label: '左髂前上棘' },
      { key: 'asiRight', label: '右髂前上棘' },
      { key: 'axillaLeft', label: '左腋窝' },
      { key: 'axillaRight', label: '右腋窝' },
    ];
    editor.replaceChildren(
      h('div', { class: 'muted', style: 'margin-bottom:8px' }, ['先在视口点选一个顶点，再点对应标记点的“吸附到所选顶点”。']),
      ...keys.map(({ key, label }) => h('div', { class: 'dab-item' }, [
        h('div', { class: 'head' }, [h('strong', {}, [label])]),
        h('div', { class: 'muted', style: 'margin-bottom:6px' }, [
          lm[key].map((v) => v.toFixed(1)).join(', '),
        ]),
        h('button', {
          class: 'ghost',
          onClick: () => {
            const mesh = workbench.cleaned ?? workbench.raw;
            if (!mesh) return;
            if (pickedVertex === null) return toast('请先在视口点选顶点', 'error');
            const snap = snapToVertex(mesh, mesh.vertex(pickedVertex));
            lm[key] = snap.position;
            workbench.landmarksValue = { ...lm };
            toast(`${label} 已移动到顶点 #${snap.vertex}`, 'success');
            renderLandmarkEditor(lm);
          },
        }, ['吸附到所选顶点']),
      ])),
    );
  }

  function existingScanList() {
    const box = h('div', {});
    const patientId = patient!.id;
    void store.listScans(patientId).then((scans) => {
      if (!scans.length) {
        box.append(h('div', { class: 'muted' }, ['该患者暂无已保存扫描']));
        return;
      }
      box.append(h('label', {}, ['历史扫描（点击重新载入）']));
      scans.forEach((s) => box.append(h('div', { class: 'dab-item' }, [
        h('div', { class: 'head' }, [
          h('strong', {}, [`第 ${s.visitIndex} 次 · ${s.filename}`]),
          h('span', { class: 'tag' }, [new Date(s.createdAt).toLocaleDateString('zh-CN')]),
        ]),
        h('div', { class: 'muted' }, [`${s.positions.length / 3} 顶点${s.synthetic ? ' · 程序化' : ''}`]),
        h('button', {
          class: 'ghost',
          style: 'margin-top:6px',
          onClick: () => {
            const mesh = new TriangleMesh(new Float32Array(s.positions), new Uint32Array(s.indices));
            workbench.reset();
            workbench.name = s.filename;
            workbench.raw = mesh;
            workbench.cleaned = mesh;
            workbench.aligned = mesh;
            workbench.landmarksValue = {
              asiLeft: s.landmarks.asiLeft,
              asiRight: s.landmarks.asiRight,
              axillaLeft: s.landmarks.axillaLeft,
              axillaRight: s.landmarks.axillaRight,
            };
            workbench.stats = s.cleaningStats ?? null;
            setBaseMesh(mesh);
            showMesh(mesh);
            refreshStatus();
            toast('历史扫描已载入', 'success');
          },
        }, ['载入修型工作台']),
      ])));
    });
    return box;
  }

  // 初始展示
  const initial = workbench.activeMesh();
  if (initial) {
    showMesh(initial);
    refreshStatus();
    if (workbench.landmarksValue) renderLandmarkEditor(workbench.landmarksValue);
  }
  try {
    gpuSession();
  } catch {
    view.stop();
  }
  return wrap;
}

function toolButton(label: string, onClick: () => void) {
  return h('button', { onClick }, [label]);
}
