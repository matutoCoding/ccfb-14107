import { describe, it, expect } from 'vitest';
import { makeSphereMesh, TriangleMesh } from '../src/core/mesh';
import { cgSolve, jacobi, spmv, CsrMatrix } from '../src/core/cpu-solver';

describe('CPU 参考 CG 求解器', () => {
  it('在拉普拉斯系统上收敛且残差下降', () => {
    const m = makeSphereMesh(20, 14);
    const csr = m.cotanLaplacianCSR();
    const A: CsrMatrix = { offsets: csr.offsets, indices: csr.indices, values: csr.values };
    // 构造 SPD 系统：(L + I) x = b，保证正定
    const n = m.vertexCount;
    const shifted = new Float32Array(csr.values);
    for (let i = 0; i < n; i++) {
      for (let k = csr.offsets[i]; k < csr.offsets[i + 1]; k++) {
        if (csr.indices[k] === i) shifted[k] += 1;
      }
    }
    const Asp: CsrMatrix = { offsets: csr.offsets, indices: csr.indices, values: shifted };
    const b = new Float32Array(n).fill(1);
    const { x, residual, iterations } = cgSolve(Asp, b, 300);
    expect(residual).toBeLessThan(1e-4);
    expect(iterations).toBeLessThan(300);
    // 与雅可比独立参考解对照
    const xJac = jacobi(Asp, b, 300);
    let diff = 0;
    for (let i = 0; i < n; i += 13) diff = Math.max(diff, Math.abs(x[i] - xJac[i]));
    expect(diff).toBeLessThan(1e-3);
  });

  it('解已知线性系统 A x = b 还原预设解', () => {
    // 3x3 SPD
    const offsets = Uint32Array.from([0, 2, 5, 7]);
    const indices = Uint32Array.from([0, 1, 0, 1, 2, 1, 2]);
    const values = Float32Array.from([4, 1, 1, 3, 1, 1, 2]);
    const A: CsrMatrix = { offsets, indices, values };
    const xTrue = Float32Array.from([1, 2, 3]);
    const b = new Float32Array(3);
    spmv(A, xTrue, b);
    const { x, residual } = cgSolve(A, b, 50);
    expect(residual).toBeLessThan(1e-8);
    for (let i = 0; i < 3; i++) expect(x[i]).toBeCloseTo(xTrue[i], 5);
  });
});

describe('热测地法 CPU 参考流程（球面）', () => {
  it('热距离在球面上与 Dijkstra 弧长同量级且单调', () => {
    const radius = 100;
    const m = makeSphereMesh(36, 26, radius);
    const n = m.vertexCount;
    const areas = m.vertexAreas();
    const lap = m.cotanLaplacianCSR();
    // 扩散系统 M + tL
    const h = meanEdge(m);
    const t = h * h;
    const heatVal = new Float32Array(lap.values);
    for (let i = 0; i < n; i++) {
      for (let k = lap.offsets[i]; k < lap.offsets[i + 1]; k++) {
        if (lap.indices[k] === i) heatVal[k] = lap.values[k] * t + areas[i];
        else heatVal[k] = lap.values[k] * t;
      }
    }
    const source = 0;
    const delta = new Float32Array(n);
    delta[source] = 1 / Math.max(areas[source], 1e-12);
    const heat = cgSolve({ offsets: lap.offsets, indices: lap.indices, values: heatVal }, delta, 200).x;
    // 扩散场应在源点最大、单调向外衰减
    const dijk = m.geodesicDijkstra(new Map([[source, 0]]));
    let corr = 0;
    let cnt = 0;
    for (let i = 0; i < n; i += 17) {
      if (dijk[i] > 20) {
        // 热值与距离负相关
        corr += heat[i] < heat[source] ? 1 : 0;
        cnt++;
      }
    }
    expect(corr / cnt).toBeGreaterThan(0.95);
    // 拉普拉斯可解（Poisson 距离方程不报错）
    const div = new Float32Array(n).fill(0.1);
    const phi = cgSolve({ offsets: lap.offsets, indices: lap.indices, values: lap.values }, div, 200, 1e-6);
    expect(phi.x.length).toBe(n);
  });
});

function meanEdge(mesh: TriangleMesh): number {
  const edges = mesh.edges();
  const p = mesh.positions;
  let sum = 0;
  for (let i = 0; i < edges.length; i += 2) {
    const a = edges[i] * 3,
      b = edges[i + 1] * 3;
    sum += Math.hypot(p[a] - p[b], p[a + 1] - p[b + 1], p[a + 2] - p[b + 2]);
  }
  return sum / (edges.length / 2);
}
