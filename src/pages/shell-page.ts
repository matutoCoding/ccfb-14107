// 支具壳体页：等距偏移、分区厚度、修剪线/开窗/后开口、倒圆、校验、STL 与修型单导出
import { h, toast, download } from '../ui/dom';
import { createViewport } from '../ui/viewport';
import { gpuSession, setShellMesh } from '../state/gpu-session';
import { store } from '../state/store';
import { workbench } from '../state/workbench';
import { TriangleMesh } from '../core/mesh';
import {
  buildShell,
  checkNormals,
  computeThicknessField,
  defaultThicknessZones,
  filletBoundary,
  ShellOptions,
  ThicknessZone,
  validateShell,
  WindowCut,
} from '../core/shell';
import { serializeSTLBinary, downloadBlob } from '../core/io-formats';
import { ShellRecord } from '../core/types';

export function shellPage(): HTMLElement {
  const baseMesh = workbench.activeMesh();
  const sculptedMesh = gpuSession().sculptedMesh;
  if (!baseMesh) return h('div', { class: 'empty' }, ['请先完成扫描处理。']);
  if (!sculptedMesh) return h('div', { class: 'empty' }, ['请先在「修型」页应用至少一笔并保存方案。']);
  const base: TriangleMesh = baseMesh;
  const sculpted: TriangleMesh = sculptedMesh;

  const wrap = h('div', { class: 'workspace wide-left' });
  const view = createViewport(620);
  view.renderer.clearMeshes();
  const side = h('div', { class: 'panel-side' });

  const bb = base.bbox();
  const settings = store.state.settings;
  const zones: ThicknessZone[] = defaultThicknessZones([bb.min[1], bb.max[1]]);
  const options: ShellOptions = {
    thicknessZones: zones,
    defaultThicknessMm: settings.defaultShellThicknessMm,
    trimLines: [],
    windows: [],
    posteriorOpeningMm: 24,
    edgeFilletMm: 3,
  };

  let shell: TriangleMesh | null = null;
  let thickness: Float32Array = new Float32Array(sculpted.vertexCount).fill(settings.defaultShellThicknessMm);
  let petg = true;

  view.renderer.removeMesh('base');
  view.renderer.removeMesh('sculpt');
  view.renderer.addMesh('inner', sculpted, { baseColor: [0.85, 0.78, 0.7], opacity: 0.25 });
  view.renderer.setPickMesh(sculpted);
  view.renderer.orbit.target = sculpted.bbox().center;

  // —— 分区厚度编辑 ——
  const zoneList = h('div', {});
  function renderZones() {
    zoneList.replaceChildren(...zones.map((z, i) => h('div', { class: 'dab-item' }, [
      h('div', { class: 'head' }, [h('strong', {}, [z.name]), h('span', { class: 'tag' }, [`${z.thicknessMm} mm`])]),
      h('div', { class: 'muted' }, [`高度区间 Y ${z.yRange[0].toFixed(0)} ~ ${z.yRange[1].toFixed(0)} mm`]),
      h('div', { class: 'row', style: 'margin-top:6px' }, [
        h('input', {
          type: 'number', min: '3', max: '5', step: '0.1', value: String(z.thicknessMm), style: 'width:90px',
          onChange: (e: Event) => {
            z.thicknessMm = Number((e.target as HTMLInputElement).value);
            renderZones();
          },
        }),
        h('button', {
          class: 'ghost danger',
          onClick: () => { zones.splice(i, 1); renderZones(); },
        }, ['删除分区']),
      ]),
    ])));
  }
  renderZones();

  // 开窗交互
  let pickedVertex: number | null = null;
  view.canvas.addEventListener('click', (e) => {
    const hit = view.renderer.pick(e.clientX, e.clientY);
    if (hit) {
      pickedVertex = hit.vertex;
      pickInfo.textContent = `已选顶点 #${hit.vertex}（可设为开窗中心或修剪线点）`;
    }
  });
  const pickInfo = h('div', { class: 'muted' }, ['点击网格选择顶点']);
  const windowRadius = h('input', { type: 'number', value: '35', min: '10', max: '80' });
  const filletInput = h('input', { type: 'number', value: '4', min: '1', max: '10' });
  const openingInput = h('input', { type: 'number', value: String(options.posteriorOpeningMm), min: '0', max: '60' });

  side.append(
    h('div', { class: 'card' }, [
      h('h3', {}, ['壳体生成']),
      h('div', { class: 'muted', style: 'margin-bottom:10px' }, [
        '内壁贴合修型面，外壁按顶点法向等距偏移 thickness(v)，边界环桥接为水密实体。',
      ]),
      h('button', {
        class: 'primary', style: 'width:100%',
        onClick: () => {
          thickness = computeThicknessField(sculpted, options);
          const t0 = performance.now();
          let built = buildShell(sculpted, thickness, options);
          built = filletBoundary(built, 2);
          shell = built;
          setShellMesh(shell);
          view.renderer.removeMesh('shell');
          view.renderer.addMesh('shell', shell, {
            baseColor: [0.55, 0.78, 0.95],
            opacity: petg ? 0.42 : 1,
            specular: 0.6,
            aoStrength: 0.6,
          });
          toast(`壳体生成完成：${shell.faceCount} 面，耗时 ${(performance.now() - t0).toFixed(0)}ms`, 'success');
          runValidation();
          checkPrinterSize();
        },
      }, ['生成等距偏移壳体']),
      h('div', { style: 'height:8px' }),
      h('div', { class: 'pill-group' }, [
        h('button', { class: 'on', onClick: (e: Event) => { petg = true; syncMat(e); } }, ['半透明 PETG']),
        h('button', { onClick: (e: Event) => { petg = false; syncMat(e); } }, ['不透明打印料']),
      ]),
    ]),
    h('div', { class: 'card' }, [
      h('h3', {}, ['分区厚度（3–5 mm）']),
      zoneList,
      h('button', {
        onClick: () => {
          zones.push({ name: `自定义分区 ${zones.length + 1}`, yRange: [bb.min[1], bb.max[1]], thetaRange: null, thicknessMm: 4 });
          renderZones();
        },
      }, ['＋ 添加整圈分区']),
    ]),
    h('div', { class: 'card' }, [
      h('h3', {}, ['修剪线 / 开窗 / 后开口']),
      pickInfo,
      field('开窗半径 mm', windowRadius),
      field('开窗倒圆半径 mm', filletInput),
      h('button', {
        style: 'width:100%;margin-bottom:8px',
        onClick: () => {
          if (pickedVertex === null) return toast('请先点击网格选择开窗中心', 'error');
          const cut: WindowCut = {
            centerVertex: pickedVertex,
            radiusMm: Number(windowRadius.value),
            filletMm: Number(filletInput.value),
          };
          options.windows.push(cut);
          toast(`已添加开窗（顶点 #${cut.centerVertex}），重新生成壳体生效`, 'success');
        },
      }, ['在所选顶点开窗（自动倒圆）']),
      field('后开口宽度 mm（0 表示不开口）', openingInput),
      h('button', {
        style: 'width:100%',
        onClick: () => {
          options.posteriorOpeningMm = Number(openingInput.value);
          toast('后开口参数已更新，重新生成壳体生效', 'success');
        },
      }, ['应用后开口宽度']),
      h('div', { class: 'muted', style: 'margin-top:8px' }, [
        '边缘修剪线：点击网格选顶点后，在其高度切出上/下缘，切口经边界倒圆平滑。',
      ]),
      h('div', { class: 'row', style: 'margin-top:8px' }, [
        h('button', {
          style: 'flex:1',
          onClick: () => {
            if (pickedVertex === null) return toast('请先点击网格选择修剪高度', 'error');
            setTrimLine('superior', pickedVertex);
          },
        }, ['设为上修剪线']),
        h('button', {
          style: 'flex:1',
          onClick: () => {
            if (pickedVertex === null) return toast('请先点击网格选择修剪高度', 'error');
            setTrimLine('inferior', pickedVertex);
          },
        }, ['设为下修剪线']),
        h('button', {
          class: 'ghost',
          onClick: () => { options.trimLines = []; toast('已清除修剪线，恢复全长', 'info'); },
        }, ['清除']),
      ]),
    ]),
    h('div', { class: 'card' }, [
      h('h3', {}, ['质量校验']),
      h('div', { id: 'validation' }, [h('div', { class: 'muted' }, ['生成壳体后自动校验'])]),
      h('div', { id: 'printer-size', style: 'margin-top:8px' }),
    ]),
    h('div', { class: 'card' }, [
      h('h3', {}, ['定稿导出']),
      h('div', { class: 'grid', style: 'gap:8px' }, [
        h('button', {
          class: 'primary',
          onClick: () => exportSTL(),
        }, ['导出 STL（3D 打印）']),
        h('button', { onClick: () => exportWorksheet() }, ['导出修型单（车间核对）']),
        h('button', { onClick: saveShellRecord }, ['壳体存入 IndexedDB']),
      ]),
    ]),
  );

  wrap.append(view.el, side);

  function syncMat(e: Event) {
    const buttons = (e.currentTarget as HTMLElement).parentElement?.querySelectorAll('button');
    buttons?.forEach((b) => b.classList.remove('on'));
    (e.currentTarget as HTMLElement).classList.add('on');
    const rm = view.renderer.getMesh('shell');
    if (rm) view.renderer.setMaterial('shell', { opacity: petg ? 0.42 : 1, specular: petg ? 0.6 : 0.3 });
  }

  function runValidation() {
    if (!shell) return;
    const report = validateShell(shell, sculpted, thickness, 300);
    const normals = checkNormals(shell);
    // 薄于 2mm / 自交高亮
    if (report.thinVertices.length || report.selfIntersectingFaces.length) {
      const colors = new Float32Array(shell.vertexCount * 3).fill(0.55, 0, 3);
      for (let i = 0; i < shell.vertexCount; i++) colors.set([0.55, 0.78, 0.95], i * 3);
      const half = sculpted.vertexCount;
      for (const v of report.thinVertices) {
        colors.set([0.95, 0.2, 0.1], v * 3);
        colors.set([0.95, 0.2, 0.1], (v + half) * 3);
      }
      view.renderer.updateColors('shell', colors);
      view.renderer.setMaterial('shell', { mode: 1 });
    } else {
      view.renderer.setMaterial('shell', { mode: 0 });
    }
    const el = document.getElementById('validation');
    el?.replaceChildren(
      checkLine('水密性', report.watertight, report.watertight ? '闭合实体' : '存在开口边界（修剪后允许）'),
      checkLine('法向一致性', normals.consistent, `${normals.flippedFaces} 个反向面`),
      checkLine('最小厚度 ≥ 2mm', report.thinVertices.length === 0, `最小 ${report.minThicknessMm.toFixed(2)}mm，${report.thinVertices.length} 个顶点过薄`),
      checkLine('自交检测', report.selfIntersectingFaces.length === 0, `${report.selfIntersectingFaces.length} 处疑似自交`),
      h('div', { class: 'check-line' }, [
        h('span', {}, ['抽点厚度误差（目标 ≤0.1mm）']),
        h('span', { class: report.thicknessSample.maxAbs <= 0.1 ? 'tag ok' : 'tag bad' }, [
          `均值 ${report.thicknessSample.mean.toFixed(3)}mm / 最大 ${report.thicknessSample.maxAbs.toFixed(3)}mm（${report.thicknessSample.samples} 点）`,
        ]),
      ]),
    );
  }

  /** 在所选顶点高度收集一整圈顶点作为修剪环 */
  function setTrimLine(edge: 'superior' | 'inferior', vertex: number) {
    const y = sculpted.positions[vertex * 3 + 1];
    const ring: number[] = [];
    const band = 6;
    for (let i = 0; i < sculpted.vertexCount; i++) {
      if (Math.abs(sculpted.positions[i * 3 + 1] - y) <= band) ring.push(i);
    }
    const existing = options.trimLines.filter((t) => t.edge !== edge);
    existing.push({ edge, vertices: ring });
    options.trimLines = existing;
    toast(`已在 Y=${y.toFixed(1)}mm 设置${edge === 'superior' ? '上' : '下'}修剪线（${ring.length} 点），重新生成壳体生效`, 'success');
  }

  function checkPrinterSize() {
    if (!shell) return;
    const b = shell.bbox();
    const size = [b.max[0] - b.min[0], b.max[1] - b.min[1], b.max[2] - b.min[2]];
    const max = settings.printerMaxSize;
    const over = [size[0] > max.x, size[1] > max.y, size[2] > max.z];
    const el = document.getElementById('printer-size');
    if (!el) return;
    if (!over.some(Boolean)) {
      el.replaceChildren(h('div', { class: 'tag ok' }, [`成型尺寸 ${size.map((v) => v.toFixed(0)).join('×')} mm，打印机可一次成型`]));
      return;
    }
    // 找超差最严重的轴作为分段方向
    const axes = ['X（冠状）', 'Y（脊柱高度）', 'Z（矢状）'];
    const worst = over.map((o, i) => (o ? size[i] / [max.x, max.y, max.z][i] : 0));
    const axis = worst.indexOf(Math.max(...worst));
    const pieces = Math.ceil(size[axis] / [max.x, max.y, max.z][axis]);
    el.replaceChildren(
      h('div', { class: 'tag bad', style: 'margin-bottom:6px' }, ['超出打印机最大成型尺寸']),
      h('div', { class: 'muted' }, [
        `建议沿 ${axes[axis]} 轴分为 ${pieces} 段，每段 ≤ ${[max.x, max.y, max.z][axis]}mm，段间留 2mm 粘接止口。`,
      ]),
    );
  }

  function exportSTL() {
    if (!shell) return toast('请先生成壳体', 'error');
    const pre = validateShell(shell, sculpted, thickness, 300);
    if (!pre.watertight) {
      if (!confirm('壳体存在开口边界（修剪/开窗所致），仍要导出吗？')) return;
    }
    const blob = serializeSTLBinary(shell);
    const patient = store.state.currentPatient;
    downloadBlob(blob, `${patient?.name ?? 'brace'}-shell-${Date.now()}.stl`);
    toast('STL 已导出', 'success');
  }

  function exportWorksheet() {
    if (!shell) return toast('请先生成壳体', 'error');
    const patient = store.state.currentPatient;
    const plan = store.state.currentPlan;
    const lines: string[] = [
      '色努式矫形支具修型单',
      '==============================',
      `患者：${patient?.name ?? '—'}    性别：${patient?.gender ?? '—'}`,
      `导出时间：${new Date().toLocaleString('zh-CN')}`,
      `关联方案：${plan?.name ?? '工作台未保存方案'}`,
      '',
      '一、压垫/释放区位置与深度',
      '------------------------------',
      ...(plan?.dabs ?? []).map((d, i) =>
        `${i + 1}. [${d.kind === 'pressure' ? '压垫' : '释放区'}] 顶点#${d.centerVertex}  σ=${d.sigmaMm}mm  深度=${d.depthMm}mm` +
        (d.contralateral?.enabled ? `  对侧空腔=${d.contralateral.depthMm}mm` : '')),
      '',
      '二、逐层截面轮廓（Y 高度 mm）',
      '------------------------------',
      ...sectionLines(),
      '',
      '三、壳体分区厚度',
      '------------------------------',
      ...zones.map((z) => `${z.name}：Y ${z.yRange[0].toFixed(0)}~${z.yRange[1].toFixed(0)}mm  厚度 ${z.thicknessMm}mm`),
      `默认厚度：${options.defaultThicknessMm}mm    后开口：${options.posteriorOpeningMm}mm`,
      '',
      '四、开孔',
      '------------------------------',
      ...options.windows.map((w) => `开窗 顶点#${w.centerVertex} 半径 ${w.radiusMm}mm 倒圆 ${w.filletMm}mm`),
      options.windows.length ? '' : '无',
      '',
      '五、校验结论',
      '------------------------------',
      (() => {
        const r = validateShell(shell, sculpted, thickness, 300);
        return [
          `水密：${r.watertight ? '通过' : '存在开口'}  最小厚度：${r.minThicknessMm.toFixed(2)}mm  最大厚度：${r.maxThicknessMm.toFixed(2)}mm`,
          `抽点厚度误差：均值 ${r.thicknessSample.mean.toFixed(3)}mm 最大 ${r.thicknessSample.maxAbs.toFixed(3)}mm`,
          `薄壳顶点：${r.thinVertices.length}  疑似自交：${r.selfIntersectingFaces.length}`,
        ].join('\n');
      })(),
      '',
      '车间技师逐项核对：□ 压垫位置  □ 压垫深度  □ 对侧空腔  □ 分区厚度  □ 修剪边缘  □ 开窗倒圆  □ 后开口',
    ];
    download(`${patient?.name ?? 'brace'}-修型单.txt`, lines.join('\n'), 'text/plain;charset=utf-8');
    toast('修型单已导出', 'success');
  }

  function sectionLines(): string[] {
    const b = sculpted.bbox();
    const levels = [0.2, 0.4, 0.6, 0.8].map((t) => b.min[1] + (b.max[1] - b.min[1]) * t);
    return levels.map((y) => {
      const ring: number[][] = [];
      for (let i = 0; i < sculpted.vertexCount; i++) {
        if (Math.abs(sculpted.positions[i * 3 + 1] - y) <= 5) {
          ring.push([sculpted.positions[i * 3], sculpted.positions[i * 3 + 2]]);
        }
      }
      const cx = ring.reduce((s, p) => s + p[0], 0) / (ring.length || 1);
      const cz = ring.reduce((s, p) => s + p[1], 0) / (ring.length || 1);
      const angles = ring.map((p) => Math.atan2(p[1] - cz, p[0] - cx)).sort((a, b2) => a - b2);
      return `Y=${y.toFixed(0)}mm：${ring.length} 个轮廓点，角向 ${angles.length} 段，周长近似 ${(ring.length * 4).toFixed(0)}mm`;
    });
  }

  async function saveShellRecord() {
    if (!shell) return toast('请先生成壳体', 'error');
    const r = validateShell(shell, sculpted, thickness, 300);
    const patient = store.state.currentPatient;
    const plan = store.state.currentPlan;
    const rec: ShellRecord = {
      id: store.newId('shell'),
      planId: plan?.id ?? 'workbench',
      patientId: patient?.id ?? 'unknown',
      createdAt: Date.now(),
      positions: shell.positions,
      indices: shell.indices,
      thicknessField: thickness,
      thicknessZones: zones,
      trimLines: options.trimLines,
      windows: options.windows,
      posteriorOpeningMm: options.posteriorOpeningMm,
      validation: {
        watertight: r.watertight,
        minThicknessMm: r.minThicknessMm,
        maxThicknessMm: r.maxThicknessMm,
        thicknessErrorMax: r.thicknessSample.maxAbs,
        thinVertexCount: r.thinVertices.length,
        selfIntersectCount: r.selfIntersectingFaces.length,
      },
    };
    await store.saveShell(rec);
    toast('壳体已存入 IndexedDB', 'success');
  }

  return wrap;
}

function field(label: string, control: HTMLElement) {
  return h('div', { class: 'field' }, [h('label', {}, [label]), control]);
}

function checkLine(label: string, ok: boolean, detail: string) {
  return h('div', { class: 'check-line' }, [
    h('span', {}, [label]),
    h('span', { class: ok ? 'tag ok' : 'tag bad' }, [detail]),
  ]);
}
