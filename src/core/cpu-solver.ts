// CPU 参考共轭梯度求解器（GPU CG 的数值对照基准）
export interface CsrMatrix {
  offsets: Uint32Array;
  indices: Uint32Array;
  values: Float32Array;
}

export function cgSolve(
  A: CsrMatrix,
  b: Float32Array,
  iterations: number,
  tol = 1e-8,
): { x: Float32Array; residual: number; iterations: number } {
  const n = b.length;
  const x = new Float32Array(n);
  const r = new Float32Array(b);
  const p = new Float32Array(b);
  const Ap = new Float32Array(n);
  let rr = dot(r, r);
  const bNorm = Math.max(1e-12, Math.sqrt(dot(b, b)));

  for (let it = 0; it < iterations; it++) {
    spmv(A, p, Ap);
    const pAp = Math.max(dot(p, Ap), 1e-20);
    const alpha = rr / pAp;
    for (let i = 0; i < n; i++) {
      x[i] += alpha * p[i];
      r[i] -= alpha * Ap[i];
    }
    const rrNew = dot(r, r);
    if (Math.sqrt(rrNew) / bNorm < tol) {
      return { x, residual: Math.sqrt(rrNew) / bNorm, iterations: it + 1 };
    }
    const beta = rrNew / rr;
    for (let i = 0; i < n; i++) p[i] = r[i] + beta * p[i];
    rr = rrNew;
  }
  return { x, residual: Math.sqrt(rr) / bNorm, iterations };
}

export function spmv(A: CsrMatrix, x: Float32Array, out: Float32Array): void {
  for (let row = 0; row < x.length; row++) {
    let sum = 0;
    for (let k = A.offsets[row]; k < A.offsets[row + 1]; k++) {
      sum += A.values[k] * x[A.indices[k]];
    }
    out[row] = sum;
  }
}

function dot(a: Float32Array, b: Float32Array): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

/** 雅可比迭代（对角占优系统的独立第二参考解） */
export function jacobi(A: CsrMatrix, b: Float32Array, iterations: number): Float32Array {
  const n = b.length;
  let x = new Float32Array(n);
  const diag = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    for (let k = A.offsets[i]; k < A.offsets[i + 1]; k++) {
      if (A.indices[k] === i) diag[i] = A.values[k];
    }
  }
  for (let it = 0; it < iterations; it++) {
    const nx = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      let off = 0;
      for (let k = A.offsets[i]; k < A.offsets[i + 1]; k++) {
        if (A.indices[k] !== i) off += A.values[k] * x[A.indices[k]];
      }
      nx[i] = (b[i] - off) / Math.max(1e-12, diag[i]);
    }
    x = nx;
  }
  return x;
}
