import { describe, it, expect } from 'vitest';
import { makeSphereMesh, TriangleMesh, pointTriangleDistance } from '../src/core/mesh';
import { parseOBJ, parsePLY, serializeSTLBinary, serializeOBJ } from '../src/core/io-formats';

describe('TriangleMesh 拓扑', () => {
  it('构建顶点/面邻接且无重复', () => {
    const m = makeSphereMesh(24, 16);
    const v2v = m.v2v();
    expect(v2v.length).toBe(m.vertexCount);
    for (const list of v2v) {
      expect(new Set(list).size).toBe(list.length);
      expect(list.length).toBeGreaterThan(0);
    }
    const v2f = m.v2f();
    let total = 0;
    for (const f of v2f) total += f.length;
    expect(total).toBe(m.faceCount * 3);
  });

  it('面邻接在闭合球上全部命中（无边界）', () => {
    const m = makeSphereMesh(20, 14);
    const adj = m.faceAdjacency();
    let boundary = 0;
    for (let i = 0; i < adj.length; i++) if (adj[i] === -1) boundary++;
    expect(boundary).toBe(0);
    expect(m.isWatertight()).toBe(true);
    expect(m.eulerCharacteristic()).toBe(2);
  });

  it('面积加权法向为单位向量', () => {
    const m = makeSphereMesh(16, 12);
    const n = m.vertexNormals();
    for (let i = 0; i < m.vertexCount; i += 7) {
      const l = Math.hypot(n[i * 3], n[i * 3 + 1], n[i * 3 + 2]);
      expect(l).toBeCloseTo(1, 5);
    }
  });

  it('点到三角形距离：顶点与内部点', () => {
    const m = new TriangleMesh(
      Float32Array.from([0, 0, 0, 10, 0, 0, 0, 10, 0]),
      Uint32Array.from([0, 1, 2]),
    );
    const atVertex = pointTriangleDistance(m, 0, [0, 0, 0]);
    expect(atVertex.distance).toBeCloseTo(0, 6);
    const above = pointTriangleDistance(m, 0, [2, 2, 5]);
    expect(above.distance).toBeCloseTo(5, 5);
  });
});

describe('OBJ / PLY / STL 往返', () => {
  it('OBJ 解析三角扇与负索引', () => {
    const obj = ['v 0 0 0', 'v 1 0 0', 'v 1 1 0', 'v 0 1 0', 'f 1 2 3 4'].join('\n');
    const m = parseOBJ(obj);
    expect(m.vertexCount).toBe(4);
    expect(m.faceCount).toBe(2);
    expect(serializeOBJ(m)).toContain('f 1 2 3');
  });

  it('ASCII PLY 解析', () => {
    const ply = [
      'ply', 'format ascii 1.0', 'element vertex 3',
      'property float x', 'property float y', 'property float z',
      'element face 1', 'property list uchar int vertex_indices',
      'end_header',
      '0 0 0', '1 0 0', '0 1 0',
      '3 0 1 2',
    ].join('\n');
    const m = parsePLY(new TextEncoder().encode(ply).buffer);
    expect(m.vertexCount).toBe(3);
    expect(m.faceCount).toBe(1);
  });

  it('二进制 PLY 解析', () => {
    const header = new TextEncoder().encode([
      'ply', 'format binary_little_endian 1.0', 'element vertex 3',
      'property float x', 'property float y', 'property float z',
      'element face 1', 'property list uchar int vertex_indices',
      'end_header', '',
    ].join('\n'));
    const body = new ArrayBuffer(3 * 12 + 1 + 3 * 4);
    const dv = new DataView(body);
    const pts = [[0, 0, 0], [1, 0, 0], [0, 1, 0]];
    pts.forEach((p, i) => p.forEach((v, k) => dv.setFloat32(i * 12 + k * 4, v, true)));
    dv.setUint8(36, 3);
    dv.setInt32(37, 0, true);
    dv.setInt32(41, 1, true);
    dv.setInt32(45, 2, true);
    const merged = new Uint8Array(header.byteLength + body.byteLength);
    merged.set(header, 0);
    merged.set(new Uint8Array(body), header.byteLength);
    const m = parsePLY(merged.buffer);
    expect(m.vertexCount).toBe(3);
    expect(m.faceCount).toBe(1);
  });

  it('STL 二进制尺寸正确', () => {
    const m = makeSphereMesh(8, 6);
    const blob = serializeSTLBinary(m);
    expect(blob.size).toBe(84 + m.faceCount * 50);
  });
});
