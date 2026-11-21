// 设置页：耐受阈值、默认壳厚、打印机成型尺寸、CG 迭代、存储用量、性能自检
import { h, toast, fmtBytes } from '../ui/dom';
import { store } from '../state/store';
import { repository } from '../core/repository';
import { generateTorso } from '../core/synthetic-torso';
import { TriangleMesh } from '../core/mesh';
import { buildShell, computeThicknessField, defaultThicknessZones, validateShell } from '../core/shell';
import { gpuSession, setBaseMesh } from '../state/gpu-session';
import { HeatGeodesic } from '../gpu/geodesic';
import { SculptEngine } from '../core/sculpt';
import { SculptDab } from '../core/sculpt-types';

export function settingsPage(): HTMLElement {
  const s = store.state.settings;
  const wrap = h('div', { class: 'grid cols-2' });

  const tolerance = h('input', { type: 'number', step: '0.1', value: String(s.skinToleranceKpa) });
  const thickness = h('input', { type: 'number', step: '0.1', min: '3', max: '5', value: String(s.defaultShellThicknessMm) });
  const sizeX = h('input', { type: 'number', value: String(s.printerMaxSize.x) });
  const sizeY = h('input', { type: 'number', value: String(s.printerMaxSize.y) });
  const sizeZ = h('input', { type: 'number', value: String(s.printerMaxSize.z) });
  const coeff = h('input', { type: 'number', step: '0.5', value: String(s.pressureCoeffKpaPerMm) });
  const cgIter = h('input', { type: 'number', min: '20', max: '300', value: String(s.cgIterations) });

  wrap.append(
    h('div', { class: 'card' }, [
      h('h3', {}, ['临床与打印参数']),
      field('皮肤耐受压力阈值 kPa', tolerance),
      field('默认壳体厚度 mm（3–5）', thickness),
      h('label', {}, ['打印机最大成型尺寸 mm']),
      h('div', { class: 'row', style: 'margin-bottom:12px' }, [
        sizedInput('X', sizeX), sizedInput('Y', sizeY), sizedInput('Z', sizeZ),
      ]),
      field('压力标定系数 kPa/mm', coeff),
      field('CG 求解迭代数（越高越准越慢）', cgIter),
      h('button', {
        class: 'primary',
        onClick: async () => {
          await store.saveSettings({
            skinToleranceKpa: Number(tolerance.value),
            defaultShellThicknessMm: Number(thickness.value),
            printerMaxSize: { x: Number(sizeX.value), y: Number(sizeY.value), z: Number(sizeZ.value) },
            pressureCoeffKpaPerMm: Number(coeff.value),
            cgIterations: Number(cgIter.value),
          });
          toast('设置已保存', 'success');
        },
      }, ['保存设置']),
      h('div', { class: 'muted', style: 'margin-top:10px' }, [
        '壳体超出成型尺寸时，壳体页会自动提示分段方向与段数；耐受阈值用于穿戴评估的超阈值点表。',
      ]),
    ]),
  );

  // 存储
  const storageCard = h('div', { class: 'card' }, [h('h3', {}, ['本机存储（IndexedDB）']), h('div', { id: 'storage' }, ['统计中…'])]);
  wrap.append(storageCard);
  void repository.estimateStorage().then(async (est) => {
    const [patients, scans, plans] = await Promise.all([
      repository.listPatients(),
      repository.listScans(),
      repository.listPlans(),
    ]);
    document.getElementById('storage')?.replaceChildren(
      h('div', { class: 'kv' }, [
        h('span', { class: 'k' }, ['患者数']), h('span', {}, [String(patients.length)]),
        h('span', { class: 'k' }, ['扫描数']), h('span', {}, [String(scans.length)]),
        h('span', { class: 'k' }, ['方案数']), h('span', {}, [String(plans.length)]),
        h('span', { class: 'k' }, ['已用空间']), h('span', {}, [fmtBytes(est.usage)]),
        h('span', { class: 'k' }, ['配额']), h('span', {}, [fmtBytes(est.quota)]),
      ]),
      h('div', { style: 'height:10px' }),
      h('div', { class: 'progress' }, [h('div', { style: `width:${Math.min(100, (est.usage / Math.max(1, est.quota)) * 100)}%` })]),
    );
  });

  // 性能自检
  const benchCard = h('div', { class: 'card' }, [
    h('h3', {}, ['性能与精度自检']),
    h('div', { class: 'muted', style: 'margin-bottom:10px' }, [
      '在约 20 万面的程序化侧弯躯干上验证：① 单笔修型 ≤200ms；② 等距偏移抽点厚度误差 ≤0.1mm；③ 热测地距离与 Dijkstra 基准对照。',
    ]),
    h('button', { id: 'bench-btn', class: 'primary' }, ['运行自检（约 10–30 秒）']),
    h('div', { id: 'bench-out', style: 'margin-top:12px' }),
  ]);
  wrap.append(benchCard);
  document.getElementById('bench-btn')?.addEventListener('click', () => runBenchmark());

  return wrap;
}

async function runBenchmark() {
  const out = document.getElementById('bench-out');
  if (!out) return;
  out.replaceChildren(h('div', { class: 'muted' }, ['生成 20 万面躯干并预热 GPU 管线…']));
  const log = (html: string, ok?: boolean) =>
    out.append(h('div', { class: 'check-line' }, [h('span', {}, [html]), ok === undefined ? h('span', {}, []) : h('span', { class: ok ? 'tag ok' : 'tag bad' }, [ok ? '通过' : '未达标'])]));

  try {
    const g = generateTorso({ cobbDeg: 35, radialSegments: 240, heightSegments: 220, holeCount: 1, seed: 51 });
    const mesh = g.mesh;
    log(`测试网格：${mesh.vertexCount} 顶点 / ${mesh.faceCount} 面`);
    const session = gpuSession();
    const geodesic = new HeatGeodesic(session.gpu.device, mesh);
    setBaseMesh(mesh);
    const engine = new SculptEngine(mesh, geodesic);
    const center = Math.floor(mesh.vertexCount * 0.55);
    const dab: SculptDab = {
      id: 'bench', kind: 'pressure', centerVertex: center, sigmaMm: 40, depthMm: 12,
      contralateral: { enabled: true, depthMm: 8, sigmaScale: 1.25 },
    };

    // 预热（首次含管线编译）
    await engine.applyPlan(new Float32Array(mesh.positions), [dab]);
    // 计时
    const t0 = performance.now();
    const result = await engine.applyPlan(new Float32Array(mesh.positions), [dab]);
    const sculptMs = performance.now() - t0;
    log(`单笔修型（含测地场+变形+对侧空腔）：${sculptMs.toFixed(0)} ms（预算 200ms）`, sculptMs <= 200);

    // 测地精度对照
    const t1 = performance.now();
    const heat = result.fields[0];
    const dijkstra = mesh.geodesicDijkstra(new Map([[center, 0]]));
    const dijkMs = performance.now() - t1;
    let errSum = 0;
    let errMax = 0;
    let cnt = 0;
    const step = Math.max(1, Math.floor(mesh.vertexCount / 2000));
    for (let i = 0; i < mesh.vertexCount; i += step) {
      if (dijkstra[i] > 5) {
        const e = Math.abs(heat[i] - dijkstra[i]) / dijkstra[i];
        errSum += e;
        errMax = Math.max(errMax, Math.abs(heat[i] - dijkstra[i]));
        cnt++;
      }
    }
    log(`测地距离：热法 vs Dijkstra 平均相对误差 ${(errSum / cnt * 100).toFixed(2)}%，最大绝对偏差 ${errMax.toFixed(2)}mm（Dijkstra 基准耗时 ${dijkMs.toFixed(0)}ms）`, errSum / cnt < 0.05);

    // 等距偏移厚度
    const sculpted = new TriangleMesh(new Float32Array(result.positions), new Uint32Array(mesh.indices));
    const tField = computeThicknessField(sculpted, {
      thicknessZones: defaultThicknessZones([sculpted.bbox().min[1], sculpted.bbox().max[1]]),
      defaultThicknessMm: store.state.settings.defaultShellThicknessMm,
      trimLines: [], windows: [], posteriorOpeningMm: 0, edgeFilletMm: 0,
    });
    const shell = buildShell(sculpted, tField, {
      thicknessZones: [], defaultThicknessMm: 4, trimLines: [], windows: [], posteriorOpeningMm: 0, edgeFilletMm: 0,
    });
    const validation = validateShell(shell, sculpted, tField, 500);
    log(`等距偏移壳体：抽点 ${validation.thicknessSample.samples} 个，厚度最大误差 ${validation.thicknessSample.maxAbs.toFixed(3)}mm（预算 0.1mm）`, validation.thicknessSample.maxAbs <= 0.1);
    log(`壳体水密：${validation.watertight ? '是' : '否'}，厚度区间 ${validation.minThicknessMm.toFixed(2)}–${validation.maxThicknessMm.toFixed(2)}mm`, validation.watertight);
    toast('自检完成', 'success');
  } catch (e) {
    log(`自检失败：${(e as Error).message}`, false);
  }
}

function field(label: string, control: HTMLElement) {
  return h('div', { class: 'field' }, [h('label', {}, [label]), control]);
}
function sizedInput(label: string, input: HTMLElement) {
  return h('div', { style: 'flex:1' }, [h('label', {}, [label]), input]);
}

void TriangleMesh;
