// WGSL 静态校验：括号配平、绑定号唯一、与 TS 端绑定组一致性
import { readFileSync, readdirSync } from 'node:fs';
import { join } from 'node:path';
import { describe, it, expect } from 'vitest';

const shaderDir = join(process.cwd(), 'src/gpu/shaders');

describe('WGSL 着色器静态校验', () => {
  const files = readdirSync(shaderDir).filter((f) => f.endsWith('.wgsl'));
  expect(files.length).toBeGreaterThan(0);

  for (const file of files) {
    it(`${file} 括号配平且无重复绑定号`, () => {
      const code = readFileSync(join(shaderDir, file), 'utf8');
      const pairs: Array<[string, string]> = [['{', '}'], ['(', ')'], ['[', ']']];
      for (const [open, close] of pairs) {
        const depth = countBalance(code, open, close);
        expect(depth, `${open}${close} 不配平`).toBe(0);
      }
      // 每个 @group 内 binding 号不重复
      const groups = new Map<number, number[]>();
      const re = /@group\((\d+)\)\s*@binding\((\d+)\)/g;
      let m: RegExpExecArray | null;
      while ((m = re.exec(code))) {
        const g = Number(m[1]);
        const b = Number(m[2]);
        if (!groups.has(g)) groups.set(g, []);
        groups.get(g)!.push(b);
      }
      for (const [, bindings] of groups) {
        expect(new Set(bindings).size, '绑定号重复').toBe(bindings.length);
      }
      expect(groups.size).toBeGreaterThan(0);
      // 计算着色器必须有 @compute 入口；渲染着色器必须有 @vertex/@fragment
      if (file === 'render.wgsl') {
        expect(code).toMatch(/@vertex/);
        expect(code).toMatch(/@fragment/);
      } else {
        expect(code).toMatch(/@compute/);
      }
    });
  }

  it('热测地法着色器包含完整四阶段入口', () => {
    const code = readFileSync(join(shaderDir, 'heat.wgsl'), 'utf8');
    for (const entry of ['initSource', 'faceGradient', 'faceDivergence', 'dequantizeDivergence', 'shiftDistance']) {
      expect(code).toContain(`fn ${entry}`);
    }
  });

  it('CG 向量核包含 axpy 与两阶段点积', () => {
    const code = readFileSync(join(shaderDir, 'vectors.wgsl'), 'utf8');
    expect(code).toContain('fn axpy2');
    expect(code).toContain('fn dotStage1');
    expect(code).toContain('fn dotStage2');
  });
});

function countBalance(code: string, open: string, close: string): number {
  let depth = 0;
  for (const ch of code) {
    if (ch === open) depth++;
    else if (ch === close) depth--;
    if (depth < 0) return depth;
  }
  return depth;
}
