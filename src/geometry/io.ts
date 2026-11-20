import { Mesh } from './mesh';

/** 解析 OBJ（支持 v / f，含 v/vt/vn 形式；多边形面自动扇形三角化） */
export function parseOBJ(text: string, name = 'scan'): Mesh {
  const positions: number[] = [];
  const indices: number[] = [];
  const lines = text.split(/\r?\n/);
  for (const raw of lines) {
    const line = raw.trim();
    if (!line || line.startsWith('#')) continue;
    const parts = line.split(/\s+/);
    if (parts[0] === 'v') {
      positions.push(+parts[1], +parts[2], +parts[3]);
    } else if (parts[0] === 'f') {
      const face: number[] = [];
      for (let k = 1; k < parts.length; k++) {
        const idx = parts[k].split('/')[0];
        let v = parseInt(idx, 10);
        if (v < 0) v = positions.length / 3 + v; // 相对索引
        else v -= 1;
        face.push(v);
      }
      for (let k = 1; k < face.length - 1; k++) {
        indices.push(face[0], face[k], face[k + 1]);
      }
    }
  }
  return {
    positions: new Float32Array(positions),
    indices: new Uint32Array(indices),
    name,
  };
}

export function serializeOBJ(mesh: Mesh): string {
  const lines: string[] = ['# Chaneau Brace Studio export'];
  for (let i = 0; i < mesh.positions.length; i += 3) {
    lines.push(`v ${mesh.positions[i].toFixed(4)} ${mesh.positions[i + 1].toFixed(4)} ${mesh.positions[i + 2].toFixed(4)}`);
  }
  for (let i = 0; i < mesh.indices.length; i += 3) {
    lines.push(`f ${mesh.indices[i] + 1} ${mesh.indices[i + 1] + 1} ${mesh.indices[i + 2] + 1}`);
  }
  return lines.join('\n');
}

/**
 * 解析 PLY（little/big endian 二进制与 ascii，每元素常见属性 x,y,z, nx,ny,nz, red,green,blue）。
 * 仅抽取三角面（或三角扇化多边形）。
 */
export function parsePLY(buffer: ArrayBuffer, name = 'scan'): Mesh {
  const bytes = new Uint8Array(buffer);
  // 读头部
  let headerEnd = -1;
  let viewStr = '';
  for (let i = 0; i < bytes.length - 10; i++) {
    if (bytes[i] === 0x65 && bytes[i + 1] === 0x64 && bytes[i + 2] === 0x0a) { // end\n
      headerEnd = i + 4;
      break;
    }
  }
  if (headerEnd < 0) throw new Error('PLY 头解析失败：找不到 end_header');
  viewStr = new TextDecoder().decode(bytes.subarray(0, headerEnd));
  const headerLines = viewStr.split(/\r?\n/);
  let format: 'ascii' | 'binary_little_endian' | 'binary_big_endian' = 'ascii';
  interface Prop { name: string; type: string; offset: number; size: number; }
  interface Element { kind: string; count: number; props: Prop[]; stride: number; }
  const elements: Element[] = [];
  let current: Element | null = null;
  const typeSize: Record<string, number> = {
    char: 1, uchar: 1, int8: 1, uint8: 1,
    short: 2, ushort: 2, int16: 2, uint16: 2,
    int: 4, uint: 4, int32: 4, uint32: 4,
    float: 4, float32: 4, double: 8, float64: 8,
  };
  for (const line of headerLines) {
    const t = line.trim().split(/\s+/);
    if (t[0] === 'format') format = t[1] as typeof format;
    else if (t[0] === 'element') {
      current = { kind: t[1], count: +t[2], props: [], stride: 0 };
      elements.push(current);
    } else if (t[0] === 'property' && current) {
      if (t[1] === 'list') {
        current.props.push({ name: '__list__', type: `${t[1]}:${t[2]}:${t[3]}:${t[4]}`, offset: -1, size: 0 });
      } else {
        const size = typeSize[t[1]] ?? 4;
        current.props.push({ name: t[2], type: t[1], offset: current.stride, size });
        current.stride += size;
      }
    }
  }
  const positions: number[] = [];
  const indices: number[] = [];

  if (format === 'ascii') {
    const body = new TextDecoder().decode(bytes.subarray(headerEnd)).split(/\r?\n/);
    let lineNo = 0;
    for (const el of elements) {
      for (let i = 0; i < el.count; i++) {
        const parts = body[lineNo++].trim().split(/\s+/);
        if (el.kind === 'vertex') {
          positions.push(+parts[0], +parts[1], +parts[2]);
        } else if (el.kind === 'face') {
          const n = +parts[0];
          for (let k = 2; k < n; k++) indices.push(+parts[1], +parts[k], +parts[k + 1]);
        }
      }
    }
  } else {
    const little = format === 'binary_little_endian';
    const dv = new DataView(buffer, headerEnd);
    let cursor = 0;
    const read = (type: string, off: number): number => {
      switch (type) {
        case 'char': case 'int8': return dv.getInt8(off);
        case 'uchar': case 'uint8': return dv.getUint8(off);
        case 'short': case 'int16': return dv.getInt16(off, little);
        case 'ushort': case 'uint16': return dv.getUint16(off, little);
        case 'int': case 'int32': return dv.getInt32(off, little);
        case 'uint': case 'uint32': return dv.getUint32(off, little);
        case 'float': case 'float32': return dv.getFloat32(off, little);
        case 'double': case 'float64': return dv.getFloat64(off, little);
        default: return dv.getFloat32(off, little);
      }
    };
    for (const el of elements) {
      for (let i = 0; i < el.count; i++) {
        if (el.kind === 'vertex') {
          const xyz = [0, 0, 0];
          for (const p of el.props) {
            if (p.name === 'x') xyz[0] = read(p.type, cursor + p.offset);
            if (p.name === 'y') xyz[1] = read(p.type, cursor + p.offset);
            if (p.name === 'z') xyz[2] = read(p.type, cursor + p.offset);
          }
          positions.push(...xyz);
          cursor += el.stride;
        } else if (el.kind === 'face') {
          const listProp = el.props.find((p) => p.name === '__list__');
          if (listProp) {
            const [, cntType, idxType] = listProp.type.split(':');
            const n = read(cntType, cursor);
            cursor += typeSize[cntType];
            const face: number[] = [];
            for (let k = 0; k < n; k++) {
              face.push(read(idxType, cursor));
              cursor += typeSize[idxType];
            }
            for (let k = 1; k < n - 1; k++) indices.push(face[0], face[k], face[k + 1]);
          } else {
            cursor += el.stride;
          }
        } else {
          cursor += el.stride;
        }
      }
    }
  }
  return {
    positions: new Float32Array(positions),
    indices: new Uint32Array(indices),
    name,
  };
}

/** 二进制 STL 导出（单位毫米，定稿给 3D 打印） */
export function serializeSTL(mesh: Mesh): ArrayBuffer {
  const triCount = mesh.indices.length / 3;
  const buffer = new ArrayBuffer(84 + triCount * 50);
  const dv = new DataView(buffer);
  const f32 = (o: number, v: number) => dv.setFloat32(o, v, true);
  // 80 字节头
  const header = new TextEncoder().encode('Chaneau Brace Studio STL');
  new Uint8Array(buffer, 0, header.length).set(header);
  dv.setUint32(80, triCount, true);
  let o = 84;
  for (let f = 0; f < triCount; f++) {
    const a = mesh.indices[f * 3] * 3;
    const b = mesh.indices[f * 3 + 1] * 3;
    const c = mesh.indices[f * 3 + 2] * 3;
    const ux = mesh.positions[b] - mesh.positions[a];
    const uy = mesh.positions[b + 1] - mesh.positions[a + 1];
    const uz = mesh.positions[b + 2] - mesh.positions[a + 2];
    const vx = mesh.positions[c] - mesh.positions[a];
    const vy = mesh.positions[c + 1] - mesh.positions[a + 1];
    const vz = mesh.positions[c + 2] - mesh.positions[a + 2];
    const nx = uy * vz - uz * vy;
    const ny = uz * vx - ux * vz;
    const nz = ux * vy - uy * vx;
    const nl = Math.hypot(nx, ny, nz) || 1;
    f32(o, nx / nl); f32(o + 4, ny / nl); f32(o + 8, nz / nl);
    o += 12;
    for (const idx of [a, b, c]) {
      f32(o, mesh.positions[idx]);
      f32(o + 4, mesh.positions[idx + 1]);
      f32(o + 8, mesh.positions[idx + 2]);
      o += 12;
    }
    o += 2; // attribute byte count
  }
  return buffer;
}

export function downloadBlob(data: BlobPart, filename: string, mime: string): void {
  const blob = data instanceof Blob ? data : new Blob([data], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 4000);
}
