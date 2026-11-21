// GPU 共轭梯度求解器（A 为 CSR 对称正定矩阵）
// 整个迭代在单个 compute pass 内完成，标量常驻 GPU，逐轮零回读
// 标量槽：0=rr 1=rrNew 2=pAp 3=alpha 4=beta
import { write } from './context';
import csrWgsl from './shaders/csr.wgsl?raw';
import vecWgsl from './shaders/vectors.wgsl?raw';

const scalarWgsl = /* wgsl */ `
struct Scalars { data: array<f32, 8> };
@group(0) @binding(0) var<storage, read_write> s: Scalars;
@compute @workgroup_size(1)
fn updateAlpha() {
  s.data[3] = s.data[0] / max(s.data[2], 1e-20);
}
@compute @workgroup_size(1)
fn updateBeta() {
  s.data[4] = s.data[1] / max(s.data[0], 1e-20);
  s.data[0] = s.data[1]; // 轮转 rr = rrNew，供下一迭代
}
`;

const broadcastWgsl = /* wgsl */ `
struct Scalars { data: array<f32, 8> };
struct U { n: f32, alpha: f32, beta: f32, slot: f32 };
@group(0) @binding(0) var<storage, read> s: Scalars;
@group(0) @binding(1) var<uniform> u: U;
@group(0) @binding(2) var<storage, read_write> out: U;
@compute @workgroup_size(1)
fn broadcastAlpha() {
  out.n = u.n;
  out.alpha = s.data[3];
  out.beta = 1.0;
  out.slot = 0.0;
}
@compute @workgroup_size(1)
fn broadcastNegAlpha() {
  out.n = u.n;
  out.alpha = 1.0;
  out.beta = -s.data[3];
  out.slot = 0.0;
}
@compute @workgroup_size(1)
fn broadcastBeta() {
  out.n = u.n;
  out.alpha = 1.0;
  out.beta = s.data[4];
  out.slot = 0.0;
}
`;

const U = GPUBufferUsage;

export interface CSRBuffers {
  offsets: GPUBuffer;
  indices: GPUBuffer;
  values: GPUBuffer;
}

interface Pool {
  x: GPUBuffer; r: GPUBuffer; p: GPUBuffer; ap: GPUBuffer; tmp: GPUBuffer;
  partial: GPUBuffer; scalars: GPUBuffer; vecUniform: GPUBuffer;
  blocks: number; n: number;
  spmvPipe: GPUComputePipeline;
  axpyPipe: GPUComputePipeline;
  dot1Pipe: GPUComputePipeline;
  dot2Pipe: GPUComputePipeline;
  copyPipe: GPUComputePipeline;
  alphaPipe: GPUComputePipeline;
  betaPipe: GPUComputePipeline;
  broadcastAlphaPipe: GPUComputePipeline;
  broadcastNegAlphaPipe: GPUComputePipeline;
  broadcastBetaPipe: GPUComputePipeline;
  csrLayout: GPUBindGroupLayout;
  vecLayout: GPUBindGroupLayout;
  scalarLayout: GPUBindGroupLayout;
  broadcastLayout: GPUBindGroupLayout;
}

export class CGSolver {
  private device: GPUDevice;
  private pools = new Map<number, Pool>();

  constructor(device: GPUDevice) {
    this.device = device;
  }

  private getPool(n: number): Pool {
    const cached = this.pools.get(n);
    if (cached) return cached;
    const device = this.device;
    const blocks = Math.ceil(n / 256);
    const vecSize = Math.ceil((n * 4) / 16) * 16;
    const mk = () => device.createBuffer({ size: vecSize, usage: U.STORAGE | U.COPY_DST | U.COPY_SRC });
    const x = mk(), r = mk(), p = mk(), ap = mk(), tmp = mk();
    const partial = device.createBuffer({ size: Math.ceil((blocks * 4) / 16) * 16, usage: U.STORAGE | U.COPY_DST });
    const scalars = device.createBuffer({ size: 32, usage: U.STORAGE | U.COPY_DST | U.COPY_SRC });
    const vecUniform = device.createBuffer({ size: 16, usage: U.UNIFORM | U.COPY_DST });

    const csrLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      ],
    });
    const vecLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      ],
    });
    const scalarLayout = device.createBindGroupLayout({
      entries: [{ binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }],
    });
    const broadcastLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      ],
    });
    const csrMod = device.createShaderModule({ code: csrWgsl });
    const vecMod = device.createShaderModule({ code: vecWgsl });
    const scalarMod = device.createShaderModule({ code: scalarWgsl });
    const broadcastMod = device.createShaderModule({ code: broadcastWgsl });
    const spmvPipe = device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [csrLayout] }),
      compute: { module: csrMod, entryPoint: 'spmv' },
    });
    const vpl = device.createPipelineLayout({ bindGroupLayouts: [vecLayout] });
    const axpyPipe = device.createComputePipeline({ layout: vpl, compute: { module: vecMod, entryPoint: 'axpy2' } });
    const dot1Pipe = device.createComputePipeline({ layout: vpl, compute: { module: vecMod, entryPoint: 'dotStage1' } });
    const dot2Pipe = device.createComputePipeline({ layout: vpl, compute: { module: vecMod, entryPoint: 'dotStage2' } });
    const copyPipe = device.createComputePipeline({ layout: vpl, compute: { module: vecMod, entryPoint: 'copyVec' } });
    const spl = device.createPipelineLayout({ bindGroupLayouts: [scalarLayout] });
    const alphaPipe = device.createComputePipeline({ layout: spl, compute: { module: scalarMod, entryPoint: 'updateAlpha' } });
    const betaPipe = device.createComputePipeline({ layout: spl, compute: { module: scalarMod, entryPoint: 'updateBeta' } });
    const bpl = device.createPipelineLayout({ bindGroupLayouts: [broadcastLayout] });
    const broadcastAlphaPipe = device.createComputePipeline({ layout: bpl, compute: { module: broadcastMod, entryPoint: 'broadcastAlpha' } });
    const broadcastNegAlphaPipe = device.createComputePipeline({ layout: bpl, compute: { module: broadcastMod, entryPoint: 'broadcastNegAlpha' } });
    const broadcastBetaPipe = device.createComputePipeline({ layout: bpl, compute: { module: broadcastMod, entryPoint: 'broadcastBeta' } });

    const pool: Pool = {
      x, r, p, ap, tmp, partial, scalars, vecUniform, blocks, n,
      spmvPipe, axpyPipe, dot1Pipe, dot2Pipe, copyPipe, alphaPipe, betaPipe,
      broadcastAlphaPipe, broadcastNegAlphaPipe, broadcastBetaPipe,
      csrLayout, vecLayout, scalarLayout, broadcastLayout,
    };
    this.pools.set(n, pool);
    return pool;
  }

  private csrGroup(pool: Pool, csr: CSRBuffers, xv: GPUBuffer, yv: GPUBuffer) {
    return this.device.createBindGroup({
      layout: pool.csrLayout,
      entries: [
        { binding: 0, resource: { buffer: pool.vecUniform } },
        { binding: 1, resource: { buffer: csr.offsets } },
        { binding: 2, resource: { buffer: csr.indices } },
        { binding: 3, resource: { buffer: csr.values } },
        { binding: 4, resource: { buffer: xv } },
        { binding: 5, resource: { buffer: yv } },
      ],
    });
  }

  private vecGroup(pool: Pool, a: GPUBuffer, b: GPUBuffer, out: GPUBuffer) {
    return this.device.createBindGroup({
      layout: pool.vecLayout,
      entries: [
        { binding: 0, resource: { buffer: pool.vecUniform } },
        { binding: 1, resource: { buffer: a } },
        { binding: 2, resource: { buffer: b } },
        { binding: 3, resource: { buffer: out } },
        { binding: 4, resource: { buffer: pool.partial } },
      ],
    });
  }

  private scalarGroup(pool: Pool) {
    return this.device.createBindGroup({
      layout: pool.scalarLayout,
      entries: [{ binding: 0, resource: { buffer: pool.scalars } }],
    });
  }

  private broadcastGroup(pool: Pool) {
    return this.device.createBindGroup({
      layout: pool.broadcastLayout,
      entries: [
        { binding: 0, resource: { buffer: pool.scalars } },
        { binding: 1, resource: { buffer: pool.vecUniform } },
        { binding: 2, resource: { buffer: pool.vecUniform } },
      ],
    });
  }

  /** 解 A x = b，返回池内 x 缓冲（复用，需保留请 copy） */
  solve(csr: CSRBuffers, b: GPUBuffer, iterations: number, initialGuess?: GPUBuffer, uniformN?: number): GPUBuffer {
    const n = uniformN ?? b.size / 4;
    const pool = this.getPool(n);
    const device = this.device;
    const { x, r, p, ap, tmp, scalars, vecUniform, blocks } = pool;
    const vCount = Math.ceil(n / 256);
    const sCount = Math.ceil(n / 64);
    const writeUniform = (alpha: number, beta: number, slot: number) =>
      write(device, vecUniform, 0, new Float32Array([n, alpha, beta, slot]));

    const dot = (pass: GPUComputePassEncoder, av: GPUBuffer, bv: GPUBuffer, slot: number) => {
      writeUniform(1, 1, slot);
      pass.setPipeline(pool.dot1Pipe);
      pass.setBindGroup(0, this.vecGroup(pool, av, bv, scalars));
      pass.dispatchWorkgroups(blocks);
      pass.setPipeline(pool.dot2Pipe);
      pass.setBindGroup(0, this.vecGroup(pool, av, bv, scalars));
      pass.dispatchWorkgroups(1);
    };

    const enc = device.createCommandEncoder();
    enc.clearBuffer(x);
    if (initialGuess) enc.copyBufferToBuffer(initialGuess, 0, x, 0, Math.min(x.size, initialGuess.size));
    enc.copyBufferToBuffer(b, 0, r, 0, r.size);
    if (initialGuess) {
      const p0 = enc.beginComputePass();
      p0.setPipeline(pool.spmvPipe);
      p0.setBindGroup(0, this.csrGroup(pool, csr, x, tmp));
      p0.dispatchWorkgroups(sCount);
      p0.end();
      writeUniform(1, -1, 0);
      const ps = enc.beginComputePass();
      ps.setPipeline(pool.axpyPipe);
      ps.setBindGroup(0, this.vecGroup(pool, r, tmp, r));
      ps.dispatchWorkgroups(vCount);
      ps.end();
    }
    enc.copyBufferToBuffer(r, 0, p, 0, p.size);

    const pass = enc.beginComputePass();
    dot(pass, r, r, 0); // rr -> slot0
    for (let it = 0; it < iterations; it++) {
      writeUniform(1, 1, 0);
      pass.setPipeline(pool.spmvPipe);
      pass.setBindGroup(0, this.csrGroup(pool, csr, p, ap));
      pass.dispatchWorkgroups(sCount);
      dot(pass, p, ap, 2); // pAp -> slot2
      pass.setPipeline(pool.alphaPipe);
      pass.setBindGroup(0, this.scalarGroup(pool));
      pass.dispatchWorkgroups(1);
      // tmp = x；广播 alpha；x = alpha*p + tmp
      writeUniform(1, 1, 0);
      pass.setPipeline(pool.copyPipe);
      pass.setBindGroup(0, this.vecGroup(pool, x, x, tmp));
      pass.dispatchWorkgroups(vCount);
      pass.setPipeline(pool.broadcastAlphaPipe);
      pass.setBindGroup(0, this.broadcastGroup(pool));
      pass.dispatchWorkgroups(1);
      pass.setPipeline(pool.axpyPipe);
      pass.setBindGroup(0, this.vecGroup(pool, p, tmp, x));
      pass.dispatchWorkgroups(vCount);
      // r = r - alpha ap（系数由 GPU 广播核写入 uniform）
      pass.setPipeline(pool.broadcastNegAlphaPipe);
      pass.setBindGroup(0, this.broadcastGroup(pool));
      pass.dispatchWorkgroups(1);
      pass.setPipeline(pool.axpyPipe);
      pass.setBindGroup(0, this.vecGroup(pool, r, ap, r));
      pass.dispatchWorkgroups(vCount);
      dot(pass, r, r, 1); // rrNew -> slot1
      pass.setPipeline(pool.betaPipe); // 同时轮转 rr
      pass.setBindGroup(0, this.scalarGroup(pool));
      pass.dispatchWorkgroups(1);
      pass.setPipeline(pool.broadcastBetaPipe);
      pass.setBindGroup(0, this.broadcastGroup(pool));
      pass.dispatchWorkgroups(1);
      // p = 1*r + beta*p
      pass.setPipeline(pool.axpyPipe);
      pass.setBindGroup(0, this.vecGroup(pool, r, p, p));
      pass.dispatchWorkgroups(vCount);
    }
    pass.end();
    device.queue.submit([enc.finish()]);
    return x;
  }
}
