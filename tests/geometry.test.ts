import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { generateTorso } from '../src/geometry/generator';
import { buildAdjacency, vertexCount, getVertex, faceCount, vertexNormals } from '../src/geometry/mesh';
import { geodesicDistance, dijkstraGeodesic } from '../src/geometry/geodesic';
import { applyStroke } from '../src/geometry/sculpt';
import { buildShell } from '../src/geometry/shell';
import { cleanScan, fillHoles, findBoundaryLoops } from '../src/geometry/process';
import { evaluateFit } from '../src/geometry/fit';
import { TriangleBVH } from '../src/geometry/bvh';
import { v3 } from '../src/geometry/math';

describe('测地距离', () => {
  const mesh = generateTorso({ segments: 80, rings: 60, cobbDeg: 30, seed: 7 });
  const adj = buildAdjacency(mesh);
  const V = vertexCount(mesh);

  test('Dijkstra 距离场非负、源点为 0、单调增长', () => {
    const src = Math.floor(V / 2);
    const { distance, method } = geodesicDistance(mesh, src);
    assert.equal(method, 'dijkstra');
    assert.equal(distance.length, V);
    assert.ok(Math.abs(distance[src]) < 1e-4);
    for (const d of distance) assert.ok(d >= -1e-5);
    const nbr = adj.neighbors[adj.offset[src]];
    const edgeLen = adj.edgeLength[adj.offset[src]];
    assert.ok(Math.abs(distance[nbr] - edgeLen) < 1e-3);
  });

  test('图距离随网格加密收敛到稳定值（测地是其上界）', () => {
    const prev = dijkstraGeodesic(
      buildAdjacency(generateTorso({ segments: 60, rings: 45, cobbDeg: 30, seed: 7 })),
      100,
    );
    const fine = dijkstraGeodesic(
      buildAdjacency(generateTorso({ segments: 120, rings: 90, cobbDeg: 30, seed: 7 })),
      100,
    );
    // 两个尺度的中位距离差异应在 15% 内（加密后趋于真实测地）
    const med = (a: Float32Array) => [...a].sort((x, y) => x - y)[a.length >> 1];
    assert.ok(Math.abs(med(prev) - med(fine)) / med(fine) < 0.15);
  });

  test('20 万面规模一笔修型在 200ms 预算内（邻接表已缓存时）', () => {
    const big = generateTorso({ segments: 160, rings: 160, cobbDeg: 35, seed: 2 });
    const adjBig = buildAdjacency(big);
    // 预热缓存
    geodesicDistance(big, 0);
    const t0 = performance.now();
    const res = applyStroke(big, adjBig, {
      kind: 'pad',
      centerVertex: Math.floor(vertexCount(big) * 0.5),
      radiusMm: 60,
      depthMm: 8,
    }, { autoContralateral: false });
    const elapsed = performance.now() - t0;
    assert.ok(elapsed < 200, `修型耗时 ${elapsed.toFixed(0)}ms 超过 200ms 预算`);
    assert.ok(res.displacement.some((d) => d < 0));
  });
});

describe('等距偏移壳厚精度', () => {
  test('随机抽点法线方向壁厚偏差 ≤ 0.1mm（含分区厚度）', () => {
    const mesh = generateTorso({ segments: 90, rings: 70, cobbDeg: 25, seed: 11 });
    const adj = buildAdjacency(mesh);
    applyStroke(mesh, adj, {
      kind: 'pad',
      centerVertex: Math.floor(vertexCount(mesh) * 0.55),
      radiusMm: 55,
      depthMm: 7,
    }, { autoContralateral: false });

    const result = buildShell(mesh, {
      zones: [
        { yMin: -360, yMax: -80, thicknessMm: 4 },
        { yMin: -80, yMax: 120, thicknessMm: 5 },
        { yMin: 120, yMax: 360, thicknessMm: 3 },
      ],
      defaultThicknessMm: 4,
      trimLines: [],
      filletRadiusMm: 0,
    });
    const { inner, outer, thicknessField } = result;
    const bvh = new TriangleBVH(outer);
    const normals = vertexNormals(mesh);
    const V = vertexCount(inner);
    let maxErr = 0;
    for (let k = 0; k < 300; k++) {
      const v = Math.floor((k * 9973 + 13) % V);
      const p = getVertex(inner, v);
      const n: [number, number, number] = [normals[v * 3], normals[v * 3 + 1], normals[v * 3 + 2]];
      const target = v3.add(p, v3.scale(n, thicknessField[v]));
      const hit = bvh.nearestPoint(target);
      // 工艺壁厚 = 内外壁命中点在法线方向的投影距离
      const measured = v3.dot(v3.sub(hit.point, p), n);
      maxErr = Math.max(maxErr, Math.abs(measured - thicknessField[v]));
    }
    assert.ok(maxErr <= 0.1, `最大壳厚误差 ${maxErr.toFixed(4)}mm 超过 0.1mm`);
    assert.ok(result.watertight, '无修剪壳体应水密');
  });

  test('同拓扑内外壁顶点法线距离严格等于厚度场', () => {
    const mesh = generateTorso({ segments: 60, rings: 45, cobbDeg: 20, seed: 4 });
    const result = buildShell(mesh, {
      zones: [], defaultThicknessMm: 4, trimLines: [], filletRadiusMm: 0,
    });
    const V = vertexCount(result.inner);
    const normals = vertexNormals(result.inner);
    let maxErr = 0;
    for (let v = 0; v < V; v += 11) {
      const p = getVertex(result.inner, v);
      const n: [number, number, number] = [normals[v * 3], normals[v * 3 + 1], normals[v * 3 + 2]];
      const q = getVertex(result.outer, v);
      const d = v3.dot(v3.sub(q, p), n);
      maxErr = Math.max(maxErr, Math.abs(d - result.thicknessField[v]));
    }
    assert.ok(maxErr < 1e-4, `同拓扑壁厚偏差 ${maxErr}`);
  });
});

describe('扫描清理与补洞', () => {
  test('破洞被识别并补齐', () => {
    const mesh = generateTorso({ segments: 80, rings: 60, holes: 3, seed: 3 });
    const loops = findBoundaryLoops(mesh);
    assert.ok(loops.length >= 3, `应检出 ≥3 个边界环，实际 ${loops.length}`);
    const before = faceCount(mesh);
    const r = fillHoles(mesh);
    assert.equal(r.holes, loops.length);
    assert.ok(faceCount(mesh) > before);
    assert.equal(findBoundaryLoops(mesh).length, 0);
  });

  test('扶架杂点连通域被清除', () => {
    const mesh = generateTorso({ segments: 70, rings: 50, withFrameJunk: true, seed: 4 });
    const v0 = vertexCount(mesh);
    const { mesh: cleaned, report } = cleanScan(mesh);
    assert.ok(report.removedComponents >= 2, '至少清掉杆+三角片连通域');
    assert.ok(vertexCount(cleaned) < v0);
  });
});

describe('穿戴评估', () => {
  test('穿透点产生正压力且可定位', () => {
    // 壳体基于原始扫描；评估时用一个被压进内壁的扫描（模拟软组织受支具压迫）
    const shellSource = generateTorso({ segments: 70, rings: 50, cobbDeg: 20, seed: 5 });
    const shell = buildShell(shellSource, {
      zones: [], defaultThicknessMm: 4, trimLines: [], filletRadiusMm: 0,
    });
    const compressedScan = generateTorso({ segments: 70, rings: 50, cobbDeg: 20, seed: 5 });
    for (let i = 0; i < compressedScan.positions.length; i++) compressedScan.positions[i] *= 0.94;
    const report = evaluateFit(compressedScan, shell.inner, 30);
    assert.ok(report.maxPenetrationMm > 0, '应检测到穿透');
    assert.ok(report.maxPressureKpa > 30, '最大压力应超阈值');
    assert.ok(report.violations.length > 0, '应列出超阈值点');
    assert.ok(report.violations[0].vertexId >= 0, '违规点可定位到顶点');
  });

  test('贴合扫描无穿透、压力为零', () => {
    const shellSource = generateTorso({ segments: 50, rings: 40, cobbDeg: 20, seed: 6 });
    const shell = buildShell(shellSource, {
      zones: [], defaultThicknessMm: 4, trimLines: [], filletRadiusMm: 0,
    });
    const sameScan = generateTorso({ segments: 50, rings: 40, cobbDeg: 20, seed: 6 });
    const report = evaluateFit(sameScan, shell.inner, 30);
    assert.ok(report.maxPenetrationMm < 0.5, '同表面应几乎无穿透');
    assert.equal(report.violations.length, 0);
  });
});

describe('BVH 最近点', () => {
  test('顶点自身查询距离≈0', () => {
    const mesh = generateTorso({ segments: 50, rings: 40, seed: 9 });
    const bvh = new TriangleBVH(mesh);
    const v = 1234;
    const hit = bvh.nearestPoint(getVertex(mesh, v));
    assert.ok(hit.distance < 1.5);
    assert.ok(hit.face >= 0);
  });
});
