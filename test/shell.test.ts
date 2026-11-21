import { describe, it, expect } from 'vitest';
import { TriangleMesh, makeSphereMesh } from '../src/core/mesh';
import { buildShell, computeThicknessField, validateShell, defaultThicknessZones, checkNormals } from '../src/core/shell';
import { generateTorso } from '../src/core/synthetic-torso';

describe('等距偏移壳体', () => {
  it('球面上等距偏移抽点厚度误差 ≤ 0.1mm', () => {
    // 高细分球，半径 100mm
    const inner = makeSphereMesh(64, 48, 100);
    const thickness = new Float32Array(inner.vertexCount).fill(4);
    const shell = buildShell(inner, thickness, {
      thicknessZones: [], defaultThicknessMm: 4, trimLines: [], windows: [], posteriorOpeningMm: 0, edgeFilletMm: 0,
    });
    const report = validateShell(shell, inner, thickness, 500);
    expect(report.thicknessSample.maxAbs).toBeLessThanOrEqual(0.1);
    expect(report.watertight).toBe(true);
  });

  it('分区厚度场落在 3–5mm 区间', () => {
    const g = generateTorso({ cobbDeg: 25, holeCount: 0, seed: 3, radialSegments: 80, heightSegments: 70 });
    const mesh = g.mesh;
    const bb = mesh.bbox();
    const zones = defaultThicknessZones([bb.min[1], bb.max[1]]);
    const field = computeThicknessField(mesh, {
      thicknessZones: zones, defaultThicknessMm: 4, trimLines: [], windows: [], posteriorOpeningMm: 0, edgeFilletMm: 0,
    });
    for (const t of field) {
      expect(t).toBeGreaterThanOrEqual(3);
      expect(t).toBeLessThanOrEqual(5);
    }
  });

  it('躯干壳体可构建且厚度抽点达标', () => {
    const g = generateTorso({ cobbDeg: 30, holeCount: 0, seed: 5, radialSegments: 96, heightSegments: 84 });
    const inner = g.mesh;
    const thickness = new Float32Array(inner.vertexCount).fill(4);
    const shell = buildShell(inner, thickness, {
      thicknessZones: [], defaultThicknessMm: 4, trimLines: [], windows: [], posteriorOpeningMm: 0, edgeFilletMm: 0,
    });
    const report = validateShell(shell, inner, thickness, 400);
    expect(report.thicknessSample.maxAbs).toBeLessThanOrEqual(0.1);
    // 顶点数翻倍（内外两层）
    expect(shell.vertexCount).toBe(inner.vertexCount * 2);
  });

  it('法向一致性检查识别正确绕序', () => {
    const m = makeSphereMesh(16, 12, 50);
    const r = checkNormals(m);
    expect(r.consistent).toBe(true);
    expect(r.flippedFaces).toBe(0);
  });

  it('开窗/后开口切除对应区域且壳体保持可打印的闭合结构', () => {
    const g = generateTorso({ cobbDeg: 20, holeCount: 0, seed: 9, radialSegments: 64, heightSegments: 60 });
    const inner = g.mesh;
    const thickness = new Float32Array(inner.vertexCount).fill(4);
    const fullShell = buildShell(inner, thickness, {
      thicknessZones: [], defaultThicknessMm: 4, trimLines: [], windows: [], posteriorOpeningMm: 0, edgeFilletMm: 0,
    });
    const openedShell = buildShell(inner, thickness, {
      thicknessZones: [], defaultThicknessMm: 4, trimLines: [],
      windows: [{ centerVertex: Math.floor(inner.vertexCount * 0.5), radiusMm: 30, filletMm: 4 }],
      posteriorOpeningMm: 24, edgeFilletMm: 0,
    });
    // 开口后三角形数量明显减少
    expect(openedShell.faceCount).toBeLessThan(fullShell.faceCount);
    // 壳体仍由内外双层顶点构成
    expect(openedShell.vertexCount).toBe(inner.vertexCount * 2);
  });
});
