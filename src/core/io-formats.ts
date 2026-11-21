// OBJ / PLY 网格导入与 STL 导出
// 仅依赖内置 API，支持顶点/面/法向，PLY 支持 ascii 与 binary_little_endian
import { TriangleMesh } from './mesh';

// ————— OBJ —————
export function parseOBJ(text: string): TriangleMesh {
  const positions: number[] = [];
  const normals: number[] = [];
  const indices: number[] = [];
  const faceVerts: number[][] = [];
  const lines = text.split(/\r?\n/);
  for (const line of lines) {
    if (line.startsWith('v ')) {
      const parts = line.trim().split(/\s+/).slice(1).map(Number);
      positions.push(parts[0], parts[1], parts[2]);
      if (parts.length >= 6) normals.push(parts[3], parts[4], parts[5]);
    } else if (line.startsWith('f ')) {
      const verts = line
        .trim()
        .split(/\s+/)
        .slice(1)
        .map((tok) => {
          const vi = tok.split('/')[0];
          const idx = Number(vi);
          return idx < 0 ? positions.length / 3 + idx : idx - 1;
        });
      faceVerts.push(verts);
    }
  }
  // 三角扇三角化（OBJ 允许多边形面）
  for (const verts of faceVerts) {
    for (let i = 1; i < verts.length - 1; i++) {
      indices.push(verts[0], verts[i], verts[i + 1]);
    }
  }
  return new TriangleMesh(Float32Array.from(positions), Uint32Array.from(indices));
}

export function serializeOBJ(mesh: TriangleMesh): string {
  const lines: string[] = ['# 色努支具工作台导出'];
  const p = mesh.positions;
  for (let i = 0; i < mesh.vertexCount; i++) {
    lines.push(`v ${p[i * 3].toFixed(4)} ${p[i * 3 + 1].toFixed(4)} ${p[i * 3 + 2].toFixed(4)}`);
  }
  for (let f = 0; f < mesh.faceCount; f++) {
    const o = f * 3;
    lines.push(`f ${mesh.indices[o] + 1} ${mesh.indices[o + 1] + 1} ${mesh.indices[o + 2] + 1}`);
  }
  return lines.join('\n');
}

// ————— PLY —————
export function parsePLY(buffer: ArrayBuffer): TriangleMesh {
  const bytes = new Uint8Array(buffer);
  // 解析头部（ASCII）
  let headerEnd = 0;
  let header = '';
  for (let i = 0; i < bytes.length - 1; i++) {
    if (bytes[i] === 10) {
      const line = new TextDecoder().decode(bytes.subarray(headerEnd, i));
      header += line + '\n';
      headerEnd = i + 1;
      if (line.trim() === 'end_header') break;
    }
  }
  const headerLines = header.split('\n').map((l) => l.trim()).filter(Boolean);
  const format = headerLines.find((l) => l.startsWith('format'))?.split(/\s+/) ?? [];
  const isAscii = format[1] === 'ascii';
  const isBinaryLE = format[1] === 'binary_little_endian';
  if (!isAscii && !isBinaryLE) throw new Error('仅支持 ASCII / binary little-endian PLY');

  interface Prop { name: string; type: string; isList: boolean; countType?: string; indexType?: string }
  const elements: Array<{ name: string; count: number; props: Prop[] }> = [];
  let current: (typeof elements)[number] | null = null;
  for (const line of headerLines) {
    const tok = line.split(/\s+/);
    if (tok[0] === 'element') {
      current = { name: tok[1], count: Number(tok[2]), props: [] };
      elements.push(current);
    } else if (tok[0] === 'property' && current) {
      if (tok[1] === 'list') {
        current.props.push({ name: tok[4], type: tok[3], isList: true, countType: tok[2], indexType: tok[3] });
      } else {
        current.props.push({ name: tok[2], type: tok[1], isList: false });
      }
    }
  }

  const positions: number[] = [];
  const indices: number[] = [];
  const typeSize: Record<string, number> = {
    char: 1, uchar: 1, short: 2, ushort: 2, int: 4, uint: 4, float: 4, double: 8,
  };
  const readScalar = (dv: DataView, off: number, type: string): number => {
    switch (type) {
      case 'char': return dv.getInt8(off);
      case 'uchar': return dv.getUint8(off);
      case 'short': return dv.getInt16(off, true);
      case 'ushort': return dv.getUint16(off, true);
      case 'int': return dv.getInt32(off, true);
      case 'uint': return dv.getUint32(off, true);
      case 'float': return dv.getFloat32(off, true);
      case 'double': return dv.getFloat64(off, true);
      default: throw new Error('未知 PLY 类型 ' + type);
    }
  };

  if (isAscii) {
    const body = new TextDecoder().decode(bytes.subarray(headerEnd)).trim().split(/\s+/);
    let cursor = 0;
    for (const el of elements) {
      for (let i = 0; i < el.count; i++) {
        const row: Record<string, number> = {};
        for (const prop of el.props) {
          if (prop.isList) {
            const cnt = Number(body[cursor++]);
            row[prop.name] = cnt;
            row[prop.name + '_items'] = body.slice(cursor, cursor + cnt).map(Number) as unknown as number;
            cursor += cnt;
          } else {
            row[prop.name] = Number(body[cursor++]);
          }
        }
        if (el.name === 'vertex') positions.push(row.x, row.y, row.z);
        if (el.name === 'face') {
          const items = (row as unknown as { vertex_indices_items?: number[] }).vertex_indices_items;
          if (items) for (let k = 1; k < items.length - 1; k++) indices.push(items[0], items[k], items[k + 1]);
        }
      }
    }
  } else {
    const dv = new DataView(buffer, headerEnd);
    let off = 0;
    for (const el of elements) {
      for (let i = 0; i < el.count; i++) {
        const row: Record<string, number | number[]> = {};
        for (const prop of el.props) {
          if (prop.isList) {
            const cnt = readScalar(dv, off, prop.countType!);
            off += typeSize[prop.countType!];
            const items: number[] = [];
            for (let k = 0; k < cnt; k++) {
              items.push(readScalar(dv, off, prop.indexType!));
              off += typeSize[prop.indexType!];
            }
            row[prop.name] = items;
          } else {
            row[prop.name] = readScalar(dv, off, prop.type);
            off += typeSize[prop.type];
          }
        }
        if (el.name === 'vertex') positions.push(row.x as number, row.y as number, row.z as number);
        if (el.name === 'face') {
          const items = row.vertex_indices as number[];
          for (let k = 1; k < items.length - 1; k++) indices.push(items[0], items[k], items[k + 1]);
        }
      }
    }
  }
  return new TriangleMesh(Float32Array.from(positions), Uint32Array.from(indices));
}

// ————— STL（定稿导出，二进制优先）—————
export function serializeSTLBinary(mesh: TriangleMesh): Blob {
  const triCount = mesh.faceCount;
  const buffer = new ArrayBuffer(84 + triCount * 50);
  const dv = new DataView(buffer);
  // 80 字节头
  const header = new TextEncoder().encode('Scoliosis brace studio STL');
  new Uint8Array(buffer, 0, header.length).set(header);
  dv.setUint32(80, triCount, true);
  const p = mesh.positions,
    idx = mesh.indices;
  let off = 84;
  for (let f = 0; f < triCount; f++) {
    const o = f * 3;
    const ia = idx[o] * 3,
      ib = idx[o + 1] * 3,
      ic = idx[o + 2] * 3;
    const ux = p[ib] - p[ia],
      uy = p[ib + 1] - p[ia + 1],
      uz = p[ib + 2] - p[ia + 2];
    const vx = p[ic] - p[ia],
      vy = p[ic + 1] - p[ia + 1],
      vz = p[ic + 2] - p[ia + 2];
    let nx = uy * vz - uz * vy;
    let ny = uz * vx - ux * vz;
    let nz = ux * vy - uy * vx;
    const nl = Math.hypot(nx, ny, nz) || 1;
    nx /= nl; ny /= nl; nz /= nl;
    dv.setFloat32(off, nx, true); dv.setFloat32(off + 4, ny, true); dv.setFloat32(off + 8, nz, true);
    off += 12;
    for (const vi of [ia, ib, ic]) {
      dv.setFloat32(off, p[vi], true);
      dv.setFloat32(off + 4, p[vi + 1], true);
      dv.setFloat32(off + 8, p[vi + 2], true);
      off += 12;
    }
    dv.setUint16(off, 0, true);
    off += 2;
  }
  return new Blob([buffer], { type: 'application/sla' });
}

export function downloadBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
