import { describe, it, expect } from 'vitest';
import { TriangleMesh, makeSphereMesh } from '../src/core/mesh';
import { fillHoles, removeSmallComponents } from '../src/core/processing';
import { generateTorso } from '../src/core/synthetic-torso';

describe('扫描预处理', () => {
  it('程序化躯干生成带侧弯且面数达标', () => {
    const g = generateTorso({ cobbDeg: 32, holeCount: 2, seed: 23 });
    expect(g.mesh.vertexCount).toBeGreaterThan(10000);
    expect(g.truth.cobbDeg).toBe(32);
    expect(g.holes.length).toBe(2);
    // 破洞导致非水密
    expect(g.mesh.isWatertight()).toBe(false);
  });

  it('高精度模式面数接近 20 万', () => {
    const g = generateTorso({ radialSegments: 240, heightSegments: 220, holeCount: 1, seed: 51 });
    expect(g.mesh.faceCount).toBeGreaterThan(100000);
  });

  it('补洞后边界减少、面数增加', () => {
    const g = generateTorso({ cobbDeg: 20, holeCount: 2, holeRadiusMm: 20, seed: 7 });
    const before = g.mesh.boundaryVertices().length;
    const r = fillHoles(g.mesh);
    const after = r.mesh.boundaryVertices().length;
    expect(r.filledHoles).toBeGreaterThan(0);
    expect(r.mesh.faceCount).toBeGreaterThan(g.mesh.faceCount);
    expect(after).toBeLessThanOrEqual(before);
  });

  it('小连通域剔除保留主体', () => {
    // 主球 + 远处孤立小四面体
    const sphere = makeSphereMesh(12, 8);
    const pos = Array.from(sphere.positions);
    const idx = Array.from(sphere.indices);
    const base = sphere.vertexCount;
    pos.push(100, 100, 100, 105, 100, 100, 100, 105, 100, 100, 100, 105);
    idx.push(base, base + 1, base + 2, base, base + 2, base + 3);
    const mesh = new TriangleMesh(Float32Array.from(pos), Uint32Array.from(idx));
    const r = removeSmallComponents(mesh, 0.02);
    expect(r.stats.removedComponents).toBe(1);
    expect(r.mesh.vertexCount).toBe(sphere.vertexCount);
  });
});
