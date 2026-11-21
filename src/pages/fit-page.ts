// 穿戴评估页：壳体套回原始扫描，间隙/穿透/压力色谱，超阈值点表点击定位
import { h, toast } from '../ui/dom';
import { createViewport } from '../ui/viewport';
import { gpuSession } from '../state/gpu-session';
import { store } from '../state/store';
import { workbench } from '../state/workbench';
import { TriangleMesh } from '../core/mesh';
import { evaluateFit, reportToVertexFields } from '../core/fit-eval';

export function fitPage(): HTMLElement {
  const baseMesh = workbench.activeMesh();
  const shellMesh = gpuSession().shellMesh;
  if (!baseMesh) return h('div', { class: 'empty' }, ['请先完成扫描处理。']);
  if (!shellMesh) return h('div', { class: 'empty' }, ['请先在「支具壳体」页生成壳体。']);
  const base: TriangleMesh = baseMesh;
  const shell: TriangleMesh = shellMesh;

  const wrap = h('div', { class: 'workspace wide-left' });
  const view = createViewport(620);
  view.renderer.clearMeshes();
  const side = h('div', { class: 'panel-side' });
  const settings = store.state.settings;

  view.renderer.removeMesh('base');
  view.renderer.removeMesh('sculpt');
  view.renderer.removeMesh('inner');
  view.renderer.removeMesh('shell');
  view.renderer.addMesh('scan', base, { baseColor: [0.8, 0.74, 0.68], opacity: 0.85, aoStrength: 0.8 });
  view.renderer.addMesh('shell', shell, { baseColor: [0.55, 0.78, 0.95], opacity: 0.35, specular: 0.6 });
  view.renderer.setPickMesh(base);
  view.renderer.orbit.target = base.bbox().center;

  const report = evaluateFit(base, shell, settings.skinToleranceKpa, settings.pressureCoeffKpaPerMm);
  const fields = reportToVertexFields(base, report);

  side.append(
    h('div', { class: 'card' }, [
      h('h3', {}, ['评估概览']),
      h('div', { class: 'grid cols-2' }, [
        metric('最大穿透', `${report.maxPenetrationMm.toFixed(2)}`, 'mm', report.maxPenetrationMm > 6),
        metric('最大间隙', `${report.maxGapMm.toFixed(2)}`, 'mm', false),
        metric('平均间隙', `${report.meanGapMm.toFixed(2)}`, 'mm', false),
        metric('接触占比', `${(report.contactAreaRatio * 100).toFixed(1)}`, '%', false),
      ]),
      h('div', { class: 'muted', style: 'margin-top:8px' }, [
        `皮肤耐受阈值 ${settings.skinToleranceKpa} kPa，压力标定 ${settings.pressureCoeffKpaPerMm} kPa/mm（可在设置页调整）`,
      ]),
    ]),
    h('div', { class: 'card' }, [
      h('h3', {}, ['色谱模式']),
      h('div', { class: 'pill-group' }, [
        h('button', { class: 'on', onClick: () => mode('gap') }, ['间隙/穿透']),
        h('button', { onClick: () => mode('pressure') }, ['接触压力']),
        h('button', { onClick: () => mode('off') }, ['关闭色谱']),
      ]),
      h('div', { class: 'legend', style: 'margin-top:10px' }, [
        h('span', {}, ['低']), h('div', { class: 'bar', style: 'background:linear-gradient(90deg,#33bf4d,#f5d938,#e63b22)' }), h('span', {}, ['高']),
      ]),
    ]),
    h('div', { class: 'card', style: 'flex:1;min-height:240px;display:flex;flex-direction:column' }, [
      h('h3', {}, [`超耐受阈值点（${report.overThreshold.length}）`]),
      h('div', { style: 'overflow:auto;flex:1' }, [
        report.overThreshold.length === 0
          ? h('div', { class: 'tag ok', style: 'padding:8px' }, ['所有接触点压力均低于皮肤耐受阈值 ✓'])
          : h('table', {}, [
              h('thead', {}, [h('tr', {}, ['顶点', '穿透 mm', '压力 kPa', '定位'].map((t) => h('th', {}, [t])))]),
              h('tbody', {}, report.overThreshold.slice(0, 200).map((s) =>
                h('tr', {}, [
                  h('td', {}, [`#${s.vertex}`]),
                  h('td', { style: 'color:var(--danger)' }, [s.gap.toFixed(2)]),
                  h('td', {}, [s.pressureKpa.toFixed(1)]),
                  h('td', {}, [h('button', {
                    class: 'ghost',
                    onClick: () => locate(s.vertex, s.position),
                  }, ['定位'])]),
                ]),
              )),
            ]),
      ]),
      report.overThreshold.length > 200 ? h('div', { class: 'muted', style: 'margin-top:6px' }, [`仅显示前 200 个，共 ${report.overThreshold.length} 个超阈值点`]) : null,
    ]),
  );

  wrap.append(view.el, side);

  function mode(m: 'gap' | 'pressure' | 'off') {
    if (m === 'off') {
      view.renderer.setMaterial('scan', { mode: 0 });
      return;
    }
    const scalar = m === 'gap' ? fields.gap : fields.pressure;
    let maxAbs = 1e-6;
    for (const v of scalar) maxAbs = Math.max(maxAbs, Math.abs(v));
    const colors = new Float32Array(base.vertexCount * 3);
    for (let i = 0; i < base.vertexCount; i++) {
      const t = Math.max(0, Math.min(1, scalar[i] / maxAbs));
      if (m === 'gap') {
        // 穿透红 -> 接触黄 -> 间隙绿
        if (scalar[i] < 0) colors.set([0.9, 0.2 + 0.6 * (1 - t), 0.15], i * 3);
        else colors.set([0.2 + 0.6 * t, 0.75 - 0.5 * t, 0.3], i * 3);
      } else {
        colors.set([0.2 + 0.7 * t, 0.75 - 0.6 * t, 0.3 - 0.2 * t], i * 3);
      }
    }
    view.renderer.updateColors('scan', colors);
    view.renderer.setMaterial('scan', { mode: 1 });
  }

  function locate(vertex: number, position: [number, number, number]) {
    // 视角聚焦到目标点：把 orbit target 移过去并推近
    view.renderer.orbit.target = position;
    view.renderer.orbit.distance = Math.min(view.renderer.orbit.distance, 320);
    // 高亮顶点：用一个小红点网格（复用扫描索引画点不可行，改为临时着色）
    const colors = new Float32Array(base.vertexCount * 3);
    for (let i = 0; i < base.vertexCount; i++) colors.set([0.8, 0.74, 0.68], i * 3);
    const adj = base.v2v()[vertex];
    const mark = (v: number) => colors.set([1, 0.9, 0.1], v * 3);
    mark(vertex);
    for (const v of adj) mark(v);
    view.renderer.updateColors('scan', colors);
    view.renderer.setMaterial('scan', { mode: 1 });
    toast(`已定位到顶点 #${vertex}（黄色高亮）`, 'info', 2200);
  }

  return wrap;
}

function metric(label: string, value: string, unit: string, warn: boolean) {
  return h('div', { class: 'card', style: 'padding:12px' }, [
    h('div', { class: 'muted', style: 'font-size:12px' }, [label]),
    h('div', { class: 'metric', style: warn ? 'color:var(--danger)' : undefined }, [
      value,
      h('small', {}, [` ${unit}`]),
    ]),
  ]);
}

void TriangleMesh;
