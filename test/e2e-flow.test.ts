import { describe, it, expect } from 'vitest';
import { generateTorso } from '../src/core/synthetic-torso';
import { fillHoles, guessLandmarks, removeSmallComponents } from '../src/core/processing';
import { SculptDab, defaultDab, extractCrossSections } from '../src/core/sculpt-types';
import { TriangleMesh } from '../src/core/mesh';
import { buildShell, computeThicknessField, defaultThicknessZones, validateShell } from '../src/core/shell';
import { evaluateFit } from '../src/core/fit-eval';
import { predictSpine } from '../src/core/spine-predict';

describe('端到端业务流程（无 GPU）', () => {
  it('扫描→处理→修型→壳体→评估→预测 全链路闭环', async () => {
    // 1) 生成带洞侧弯扫描
    const g = generateTorso({ cobbDeg: 32, holeCount: 2, seed: 23, radialSegments: 64, heightSegments: 56 });
    expect(g.mesh.isWatertight()).toBe(false);

    // 2) 去杂点 + 补洞
    const cleaned = removeSmallComponents(g.mesh).mesh;
    const filled = fillHoles(cleaned);
    expect(filled.filledHoles).toBeGreaterThan(0);

    // 3) 标记点识别
    const lm = guessLandmarks(filled.mesh);
    expect(lm.asiLeft).toHaveLength(3);
    expect(lm.axillaRight).toHaveLength(3);

    // 4) 修型（CPU 路径：手工构造测地场，避免 GPU 依赖）
    const mesh = filled.mesh;
    const normals = mesh.vertexNormals();
    const center = Math.floor(mesh.vertexCount * 0.5);
    // 用 Dijkstra 距离作为测地场，手工应用一个压垫
    const geo = mesh.geodesicDijkstra(new Map([[center, 0]]));
    const base = new Float32Array(mesh.positions);
    const sculpted = new Float32Array(base);
    const dab: SculptDab = {
      ...defaultDab(center, 'pressure'),
      contralateral: { enabled: false, depthMm: 0, sigmaScale: 1 },
    };
    for (let i = 0; i < mesh.vertexCount; i++) {
      const w = Math.exp(-(geo[i] ** 2) / (2 * dab.sigmaMm ** 2));
      const disp = -dab.depthMm * w;
      sculpted[i * 3] += normals[i * 3] * disp;
      sculpted[i * 3 + 1] += normals[i * 3 + 1] * disp;
      sculpted[i * 3 + 2] += normals[i * 3 + 2] * disp;
    }
    // 修型确实改变了表面
    let maxDisp = 0;
    for (let i = 0; i < mesh.vertexCount; i++) {
      maxDisp = Math.max(maxDisp, Math.hypot(
        sculpted[i * 3] - base[i * 3],
        sculpted[i * 3 + 1] - base[i * 3 + 1],
        sculpted[i * 3 + 2] - base[i * 3 + 2],
      ));
    }
    expect(maxDisp).toBeGreaterThan(5);
    expect(maxDisp).toBeLessThanOrEqual(dab.depthMm + 0.5);

    // 5) 截面提取
    const bb = mesh.bbox();
    const sections = extractCrossSections(mesh, sculpted, [bb.min[1] + bb.max[1] * 0.3], 6);
    expect(sections.length).toBe(1);
    expect(sections[0].before.length).toBeGreaterThan(3);

    // 6) 壳体
    const sculptedMesh = new TriangleMesh(sculpted, new Uint32Array(mesh.indices));
    const thickness = computeThicknessField(sculptedMesh, {
      thicknessZones: defaultThicknessZones([bb.min[1], bb.max[1]]),
      defaultThicknessMm: 4, trimLines: [], windows: [], posteriorOpeningMm: 0, edgeFilletMm: 0,
    });
    const shell = buildShell(sculptedMesh, thickness, {
      thicknessZones: [], defaultThicknessMm: 4, trimLines: [], windows: [], posteriorOpeningMm: 0, edgeFilletMm: 0,
    });
    const report = validateShell(shell, sculptedMesh, thickness, 200);
    expect(report.watertight).toBe(true);
    expect(report.thicknessSample.maxAbs).toBeLessThanOrEqual(0.1);

    // 7) 穿戴评估
    const fit = evaluateFit(mesh, shell, 26.7, 12);
    expect(fit.samples.length).toBeGreaterThan(0);
    expect(Number.isFinite(fit.meanGapMm)).toBe(true);

    // 8) 脊柱预测
    const pred = predictSpine(mesh, [dab], 32);
    expect(pred.originalCobbDeg).toBe(32);
    expect(pred.correctedCurve.length).toBe(pred.originalCurve.length);
    expect(pred.predictedCobbDeg).toBeGreaterThanOrEqual(0);
  });

  it('默认压垫带对侧空腔配置', () => {
    const d = defaultDab(10, 'pressure');
    expect(d.contralateral?.enabled).toBe(true);
    expect(d.contralateral!.depthMm).toBeGreaterThan(0);
    const r = defaultDab(10, 'relief');
    expect(r.contralateral).toBeUndefined();
  });
});
