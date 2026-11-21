// GPU 穿戴评估核：扫描点到壳体最近面的带符号距离与接触压力
import { TriangleMesh } from '../core/mesh';
import { createStorageBuffer, write } from './context';
import fitWgsl from './shaders/fit.wgsl?raw';

const U = GPUBufferUsage;

interface UniformGrid {
  cells: { start: number; count: number }[];
  cellFaces: Uint32Array;
  origin: [number, number, number];
  cellSize: number;
  dim: [number, number, number];
}

/** 在 CPU 侧构建壳体三角形的均匀网格（桶排序），上传 GPU */
function buildGrid(shell: TriangleMesh, targetBuckets = 64): UniformGrid {
  const bb = shell.bbox();
  const span: [number, number, number] = [
    bb.max[0] - bb.min[0],
    bb.max[1] - bb.min[1],
    bb.max[2] - bb.min[2],
  ];
  const volume = Math.max(1e-6, span[0] * span[1] * span[2]);
  const cellSize = Math.cbrt((shell.faceCount / targetBuckets) * volume / Math.max(1, shell.faceCount));
  const dim: [number, number, number] = [
    Math.max(1, Math.ceil(span[0] / cellSize) + 1),
    Math.max(1, Math.ceil(span[1] / cellSize) + 1),
    Math.max(1, Math.ceil(span[2] / cellSize) + 1),
  ];
  const buckets: number[][] = Array.from({ length: dim[0] * dim[1] * dim[2] }, () => []);
  const idx = (i: number, j: number, k: number) => (i * dim[1] + j) * dim[2] + k;
  const p = shell.positions;
  for (let f = 0; f < shell.faceCount; f++) {
    const o = f * 3;
    let minX = Infinity, minY = Infinity, minZ = Infinity;
    let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
    for (let k = 0; k < 3; k++) {
      const vi = idx0(o + k);
      minX = Math.min(minX, p[vi]); maxX = Math.max(maxX, p[vi]);
      minY = Math.min(minY, p[vi + 1]); maxY = Math.max(maxY, p[vi + 1]);
      minZ = Math.min(minZ, p[vi + 2]); maxZ = Math.max(maxZ, p[vi + 2]);
    }
    const i0 = Math.floor((minX - bb.min[0]) / cellSize), i1 = Math.floor((maxX - bb.min[0]) / cellSize);
    const j0 = Math.floor((minY - bb.min[1]) / cellSize), j1 = Math.floor((maxY - bb.min[1]) / cellSize);
    const k0 = Math.floor((minZ - bb.min[2]) / cellSize), k1 = Math.floor((maxZ - bb.min[2]) / cellSize);
    for (let i = i0; i <= i1; i++)
      for (let j = j0; j <= j1; j++)
        for (let k = k0; k <= k1; k++) buckets[idx(i, j, k)].push(f);
  }
  const cells: { start: number; count: number }[] = [];
  const cellFaces: number[] = [];
  for (const bucket of buckets) {
    cells.push({ start: cellFaces.length, count: bucket.length });
    cellFaces.push(...bucket);
  }
  return { cells, cellFaces: Uint32Array.from(cellFaces), origin: bb.min, cellSize, dim };
  function idx0(local: number) {
    return shell.indices[local] * 3;
  }
}

export interface GpuFitResult {
  gap: GPUBuffer;
  pressure: GPUBuffer;
  nearestFace: GPUBuffer;
  count: number;
}

export class FitKernel {
  private device: GPUDevice;
  private pipe: GPUComputePipeline;
  private layout: GPUBindGroupLayout;
  private params: GPUBuffer;
  private gridOrigin: GPUBuffer;
  private gridDim: GPUBuffer;

  constructor(device: GPUDevice) {
    this.device = device;
    this.params = device.createBuffer({ size: 16, usage: U.UNIFORM | U.COPY_DST });
    this.gridOrigin = device.createBuffer({ size: 16, usage: U.UNIFORM | U.COPY_DST });
    this.gridDim = device.createBuffer({ size: 16, usage: U.UNIFORM | U.COPY_DST });
    this.layout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        ...[1, 2, 3, 4, 5, 6].map((b) => ({
          binding: b,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: 'read-only-storage' as GPUBufferBindingType },
        })),
        { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        ...[9, 10, 11].map((b) => ({
          binding: b, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' as GPUBufferBindingType },
        })),
      ],
    });
    const mod = device.createShaderModule({ code: fitWgsl });
    this.pipe = device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.layout] }),
      compute: { module: mod, entryPoint: 'fitDistance' },
    });
  }

  evaluate(scan: TriangleMesh, shell: TriangleMesh, shellNormals: Float32Array, toleranceKpa: number): GpuFitResult {
    const device = this.device;
    const grid = buildGrid(shell);
    const n = scan.vertexCount;
    const cellStruct = new Float32Array(grid.cells.length * 2);
    grid.cells.forEach((c, i) => {
      cellStruct[i * 2] = c.start;
      cellStruct[i * 2 + 1] = c.count;
    });
    const scanPos = createStorageBuffer(device, scan.positions);
    const shellPos = createStorageBuffer(device, shell.positions);
    const shellIdx = createStorageBuffer(device, shell.indices);
    const shellNrm = createStorageBuffer(device, shellNormals);
    const cellsBuf = createStorageBuffer(device, cellStruct);
    const cellFacesBuf = createStorageBuffer(device, grid.cellFaces);
    const gapBuf = device.createBuffer({ size: Math.ceil((n * 4) / 16) * 16, usage: U.STORAGE | U.COPY_SRC });
    const pressureBuf = device.createBuffer({ size: Math.ceil((n * 4) / 16) * 16, usage: U.STORAGE | U.COPY_SRC });
    const faceBuf = device.createBuffer({ size: Math.ceil((n * 4) / 16) * 16, usage: U.STORAGE | U.COPY_SRC });

    write(device, this.params, 0, new Float32Array([n, shell.faceCount, grid.cells.length, toleranceKpa]));
    write(device, this.gridOrigin, 0, new Float32Array([grid.origin[0], grid.origin[1], grid.origin[2], grid.cellSize]));
    write(device, this.gridDim, 0, new Uint32Array([grid.dim[0], grid.dim[1], grid.dim[2], 0]));

    const bg = device.createBindGroup({
      layout: this.layout,
      entries: [
        { binding: 0, resource: { buffer: this.params } },
        { binding: 1, resource: { buffer: scanPos } },
        { binding: 2, resource: { buffer: shellPos } },
        { binding: 3, resource: { buffer: shellIdx } },
        { binding: 4, resource: { buffer: shellNrm } },
        { binding: 5, resource: { buffer: cellsBuf } },
        { binding: 6, resource: { buffer: cellFacesBuf } },
        { binding: 7, resource: { buffer: this.gridOrigin } },
        { binding: 8, resource: { buffer: this.gridDim } },
        { binding: 9, resource: { buffer: gapBuf } },
        { binding: 10, resource: { buffer: pressureBuf } },
        { binding: 11, resource: { buffer: faceBuf } },
      ],
    });
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(this.pipe);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(Math.ceil(n / 64));
    pass.end();
    device.queue.submit([enc.finish()]);
    return { gap: gapBuf, pressure: pressureBuf, nearestFace: faceBuf, count: n };
  }
}
