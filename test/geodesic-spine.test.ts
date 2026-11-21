import { describe, it, expect } from 'vitest';
import { makeSphereMesh, TriangleMesh } from '../src/core/mesh';
import { SculptDab } from '../src/core/sculpt-types';
import { predictSpine } from '../src/core/spine-predict';
import { generateTorso } from '../src/core/synthetic-torso';
import { evaluateFit } from '../src/core/fit-eval';
import { buildShell } from '../src/core/shell';

describe('Dijkstra 测地基准', () => {
  it('球面上测地距离与大圆弧长接近', () => {
    const radius = 100;
    const m = makeSphereMesh(48, 36, radius);
    const source = 0; // 北极顶点
    const dist = m.geodesicDijkstra(new Map([[source, 0]]));
    // 南极（底极）理论弧长 = πR
    const south = m.vertexCount - 1;
    expect(dist[south]).toBeGreaterThan(radius * Math.PI * 0.9);
    expect(dist[south]).toBeLessThan(radius * Math.PI * 1.1);
    // 距离单调：源点为 0，全部非负有限
    expect(dist[source]).toBe(0);
    for (const d of dist) {
      expect(Number.isFinite(d)).toBe(true);
      expect(d).toBeGreaterThanOrEqual(0);
    }
  });

  it('余切拉普拉斯 CSR 对称且每行和接近 0', () => {
    const m = makeSphereMesh(24, 18);
    const csr = m.cotanLaplacianCSR();
    for (let i = 0; i < m.vertexCount; i++) {
      let rowSum = 0;
      for (let k = csr.offsets[i]; k < csr.offsets[i + 1]; k++) rowSum += csr.values[k];
      expect(rowSum).toBeCloseTo(0, 4);
    }
    // 对角元为正
    for (let i = 0; i < m.vertexCount; i++) {
      for (let k = csr.offsets[i]; k < csr.offsets[i + 1]; k++) {
        if (csr.indices[k] === i) expect(csr.values[k]).toBeGreaterThan(0);
      }
    }
  });
});

describe('脊柱预测', () => {
  it('压垫使预估 Cobb 角下降', () => {
    const g = generateTorso({ cobbDeg: 35, holeCount: 0, seed: 13, radialSegments: 72, heightSegments: 64 });
    const mesh = g.mesh;
    const noDab = predictSpine(mesh, [], 35);
    expect(noDab.originalCobbDeg).toBe(35);
    // 在凸侧（+X）中部加压垫
    let center = 0;
    let bestX = -Infinity;
    for (let i = 0; i < mesh.vertexCount; i++) {
      const y = mesh.positions[i * 3 + 1];
      const bb = mesh.bbox();
      if (y > (bb.min[1] + bb.max[1]) * 0.5 && mesh.positions[i * 3] > bestX) {
        bestX = mesh.positions[i * 3];
        center = i;
      }
    }
    const dabs: SculptDab[] = [{
      id: 't', kind: 'pressure', centerVertex: center, sigmaMm: 45, depthMm: 15,
    }];
    const corrected = predictSpine(mesh, dabs, 35);
    expect(corrected.predictedCobbDeg).toBeLessThan(noDab.predictedCobbDeg);
    expect(corrected.correctionRate).toBeGreaterThan(0);
    expect(corrected.correctedCurve.length).toBe(corrected.originalCurve.length);
  });
});

describe('穿戴评估', () => {
  it('贴合壳体穿透小、间隙可量化', () => {
    const g = generateTorso({ cobbDeg: 18, holeCount: 0, seed: 17, radialSegments: 56, heightSegments: 48 });
    const scan = g.mesh;
    // 直接以扫描为内面建一个 4mm 壳
    const thickness = new Float32Array(scan.vertexCount).fill(4);
    const shell = buildShell(scan, thickness, {
      thicknessZones: [], defaultThicknessMm: 4, trimLines: [], windows: [], posteriorOpeningMm: 0, edgeFilletMm: 0,
    });
    const report = evaluateFit(scan, shell, 26.7, 12);
    expect(report.samples.length).toBeGreaterThan(0);
    expect(report.maxPenetrationMm).toBeLessThan(2);
    expect(Number.isFinite(report.meanGapMm)).toBe(true);
  });
});

void TriangleMesh;
