// 校验 WGSL 存储缓冲的读写修饰符与 TS 端 BindGroupLayout 类型一致
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { describe, it, expect } from 'vitest';

const shaderDir = join(process.cwd(), 'src/gpu/shaders');

/** 解析 WGSL：返回 Map<"group:binding", 'storage' | 'read-only-storage' | 'uniform'> */
function parseWgslStorage(code: string): Map<string, string> {
  const out = new Map<string, string>();
  // 匹配 @group(g) @binding(b) var<storage, read_write|read> 或 var<uniform>
  const re = /@group\((\d+)\)\s*@binding\((\d+)\)\s*var(?:<([^>]+)>)?/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(code))) {
    const key = `${m[1]}:${m[2]}`;
    const qualifier = m[3] ?? '';
    if (qualifier.startsWith('uniform')) out.set(key, 'uniform');
    else if (qualifier.includes('atomic')) out.set(key, 'storage'); // atomic 必须读写
    else if (qualifier.includes('read_write')) out.set(key, 'storage');
    else if (qualifier.includes('read')) out.set(key, 'read-only-storage');
  }
  return out;
}

describe('WGSL 与 TS 绑定布局读写一致性', () => {
  const cases: Array<{ wgsl: string; ts: string }> = [
    { wgsl: 'heat.wgsl', ts: join(process.cwd(), 'src/gpu/geodesic.ts') },
    { wgsl: 'geometry.wgsl', ts: join(process.cwd(), 'src/gpu/geometry-kernel.ts') },
    { wgsl: 'fit.wgsl', ts: join(process.cwd(), 'src/gpu/fit-kernel.ts') },
  ];

  for (const { wgsl, ts } of cases) {
    it(`${wgsl} 的 read_write 缓冲在 TS 布局里不是 read-only-storage`, () => {
      const wgslMap = parseWgslStorage(readFileSync(join(shaderDir, wgsl), 'utf8'));
      const tsCode = readFileSync(ts, 'utf8');
      for (const [key, type] of wgslMap) {
        if (type !== 'storage') continue; // 只核对必须可写的绑定
        const [, binding] = key.split(':');
        // TS 端该 binding 不能被硬编码为 read-only-storage
        const re = new RegExp(`binding:\\s*${binding}[^}]*read-only-storage`);
        expect(re.test(tsCode), `${wgsl} binding ${binding} 应为可写 storage`).toBe(false);
      }
    });
  }

  it('所有计算着色器的存储绑定都能被解析到', () => {
    for (const f of ['heat.wgsl', 'geometry.wgsl', 'fit.wgsl', 'deform.wgsl', 'csr.wgsl', 'vectors.wgsl']) {
      const map = parseWgslStorage(readFileSync(join(shaderDir, f), 'utf8'));
      expect(map.size, `${f} 未解析到任何绑定`).toBeGreaterThan(0);
    }
  });
});
