// 热测地法高层封装：输入网格 + 源顶点，输出每顶点测地距离（mm）
// 精度对照：CPU Dijkstra 边长图距离作为基准，误差控制在网格分辨率量级
import { TriangleMesh } from '../core/mesh';
import { CGSolver, CSRBuffers } from './cg-solver';
import { createStorageBuffer, readBufferF32, write } from './context';
import heatWgsl from './shaders/heat.wgsl?raw';

const U = GPUBufferUsage;

export interface GeodesicResult {
  distance: Float32Array;
  /** GPU 总耗时 ms（不含回读） */
  gpuMs: number;
}

export class HeatGeodesic {
  private device: GPUDevice;
  private cg: CGSolver;
  private mesh: TriangleMesh;
  private n: number;
  private f: number;

  // 静态网格资源
  private posBuf: GPUBuffer;
  private idxBuf: GPUBuffer;
  private vAreaBuf: GPUBuffer;
  private fAreaBuf: GPUBuffer;
  private laplacian: CSRBuffers;
  private heatSystem: CSRBuffers;

  // 动态缓冲
  private heatBuf: GPUBuffer;
  private gradX: GPUBuffer;
  private gradY: GPUBuffer;
  private gradZ: GPUBuffer;
  private divBuf: GPUBuffer;
  private divInt: GPUBuffer;
  private distBuf: GPUBuffer;
  private uniform: GPUBuffer;
  private pipes: Record<string, GPUComputePipeline>;
  private layout: GPUBindGroupLayout;
  private pipeLayout: GPUPipelineLayout;

  constructor(device: GPUDevice, mesh: TriangleMesh) {
    this.device = device;
    this.mesh = mesh;
    this.n = mesh.vertexCount;
    this.f = mesh.faceCount;
    this.cg = new CGSolver(device);

    const vAreas = mesh.vertexAreas();
    const fAreas = mesh.faceAreas();
    this.posBuf = createStorageBuffer(device, mesh.positions);
    this.idxBuf = createStorageBuffer(device, mesh.indices);
    this.vAreaBuf = createStorageBuffer(device, vAreas);
    this.fAreaBuf = createStorageBuffer(device, fAreas);

    // L（余切拉普拉斯）与 A = M + tL（隐式扩散系统）
    const lap = mesh.cotanLaplacianCSR();
    this.laplacian = {
      offsets: createStorageBuffer(device, lap.offsets),
      indices: createStorageBuffer(device, lap.indices),
      values: createStorageBuffer(device, lap.values),
    };
    // 扩散时间 t = h²（h 为平均边长），热法经典取值
    const h = meanEdgeLength(mesh);
    const t = h * h;
    const heatValues = new Float32Array(lap.values);
    for (let i = 0; i < this.n; i++) {
      for (let k = lap.offsets[i]; k < lap.offsets[i + 1]; k++) {
        if (lap.indices[k] === i) heatValues[k] = lap.values[k] * t + vAreas[i];
        else heatValues[k] = lap.values[k] * t;
      }
    }
    this.heatSystem = {
      offsets: this.laplacian.offsets,
      indices: this.laplacian.indices,
      values: createStorageBuffer(device, heatValues),
    };

    const vecSize = Math.ceil((this.n * 4) / 16) * 16;
    const faceSize = Math.ceil((this.f * 4) / 16) * 16;
    this.heatBuf = device.createBuffer({ size: vecSize, usage: U.STORAGE | U.COPY_DST | U.COPY_SRC });
    this.gradX = device.createBuffer({ size: faceSize, usage: U.STORAGE | U.COPY_DST });
    this.gradY = device.createBuffer({ size: faceSize, usage: U.STORAGE | U.COPY_DST });
    this.gradZ = device.createBuffer({ size: faceSize, usage: U.STORAGE | U.COPY_DST });
    this.divBuf = device.createBuffer({ size: vecSize, usage: U.STORAGE | U.COPY_DST });
    this.divInt = device.createBuffer({ size: vecSize, usage: U.STORAGE | U.COPY_DST });
    this.distBuf = device.createBuffer({ size: vecSize, usage: U.STORAGE | U.COPY_DST | U.COPY_SRC });
    this.uniform = device.createBuffer({ size: 16, usage: U.UNIFORM | U.COPY_DST });

    // 绑定类型须与 heat.wgsl 声明一致：1/2/3/10 只读，4-9 读写，11 原子读写
    const readWrite = new Set([4, 5, 6, 7, 8, 9, 11]);
    this.layout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        ...[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11].map((b) => ({
          binding: b,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: (readWrite.has(b) ? 'storage' : 'read-only-storage') as GPUBufferBindingType },
        })),
      ],
    });
    this.pipeLayout = device.createPipelineLayout({ bindGroupLayouts: [this.layout] });
    const mod = device.createShaderModule({ code: heatWgsl });
    const mk = (entry: string) =>
      device.createComputePipeline({ layout: this.pipeLayout, compute: { module: mod, entryPoint: entry } });
    this.pipes = {
      initSource: mk('initSource'),
      faceGradient: mk('faceGradient'),
      faceDivergence: mk('faceDivergence'),
      dequantize: mk('dequantizeDivergence'),
      shift: mk('shiftDistance'),
    };
  }

  private heatBindGroup(): GPUBindGroup {
    return this.device.createBindGroup({
      layout: this.layout,
      entries: [
        { binding: 0, resource: { buffer: this.uniform } },
        { binding: 1, resource: { buffer: this.posBuf } },
        { binding: 2, resource: { buffer: this.idxBuf } },
        { binding: 3, resource: { buffer: this.vAreaBuf } },
        { binding: 4, resource: { buffer: this.heatBuf } },
        { binding: 5, resource: { buffer: this.gradX } },
        { binding: 6, resource: { buffer: this.gradY } },
        { binding: 7, resource: { buffer: this.gradZ } },
        { binding: 8, resource: { buffer: this.divBuf } },
        { binding: 9, resource: { buffer: this.distBuf } },
        { binding: 10, resource: { buffer: this.fAreaBuf } },
        { binding: 11, resource: { buffer: this.divInt } },
      ],
    });
  }

  /**
   * 计算到 sourceVertex 的测地距离场
   * @param cgIterations 每个线性系统的 CG 迭代数（默认 120，20 万面足够收敛）
   */
  async solve(sourceVertex: number, cgIterations = 120): Promise<GeodesicResult> {
    const device = this.device;
    const t0 = performance.now();
    // 源标记：-1 表示源顶点
    const marker = new Float32Array(this.n);
    marker[sourceVertex] = -1;
    write(device, this.heatBuf, 0, marker);
    write(device, this.uniform, 0, new Float32Array([this.n, this.f, 0, 0]));

    const bg = this.heatBindGroup();
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(this.pipes.initSource);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(Math.ceil(this.n / 64));
    pass.end();
    device.queue.submit([enc.finish()]);

    // 1) 隐式扩散 (M+tL) u = delta
    const u = this.cg.solve(this.heatSystem, this.heatBuf, cgIterations);
    device.queue.submit([]); // 确保排序
    // CG 返回池内缓冲，拷贝到 heatBuf
    const enc2 = device.createCommandEncoder();
    enc2.copyBufferToBuffer(u, 0, this.heatBuf, 0, this.heatBuf.size);
    device.queue.submit([enc2.finish()]);

    // 2) 单位化负面梯度 + 3) 散度
    const enc3 = device.createCommandEncoder();
    const pass3 = enc3.beginComputePass();
    pass3.setPipeline(this.pipes.faceGradient);
    pass3.setBindGroup(0, this.heatBindGroup());
    pass3.dispatchWorkgroups(Math.ceil(this.f / 64));
    pass3.setPipeline(this.pipes.faceDivergence);
    pass3.setBindGroup(0, this.heatBindGroup());
    pass3.dispatchWorkgroups(Math.ceil(this.f / 64));
    pass3.setPipeline(this.pipes.dequantize);
    pass3.setBindGroup(0, this.heatBindGroup());
    pass3.dispatchWorkgroups(Math.ceil(this.n / 64));
    pass3.end();
    device.queue.submit([enc3.finish()]);

    // 4) Poisson：L phi = div（符号取负与距离方向匹配）
    const phi = this.cg.solve(this.laplacian, this.divBuf, cgIterations);
    const enc4 = device.createCommandEncoder();
    enc4.copyBufferToBuffer(phi, 0, this.distBuf, 0, this.distBuf.size);
    device.queue.submit([enc4.finish()]);

    // 5) 平移使最小距离为 0
    const phiValues = await readBufferF32(device, this.distBuf, this.n);
    const minDist = Math.min(...phiValues);
    write(device, this.uniform, 8, new Float32Array([minDist]));
    const enc5 = device.createCommandEncoder();
    const pass5 = enc5.beginComputePass();
    pass5.setPipeline(this.pipes.shift);
    pass5.setBindGroup(0, this.heatBindGroup());
    pass5.dispatchWorkgroups(Math.ceil(this.n / 64));
    pass5.end();
    device.queue.submit([enc5.finish()]);

    const distance = await readBufferF32(device, this.distBuf, this.n);
    return { distance, gpuMs: performance.now() - t0 };
  }

  /** 多源距离场（对侧空腔用），sources 为顶点号集合 */
  async solveMulti(sources: number[], cgIterations = 120): Promise<GeodesicResult> {
    if (sources.length === 1) return this.solve(sources[0], cgIterations);
    // 多源：初场全部标源，热法天然支持（delta 为多源之和）
    const device = this.device;
    const marker = new Float32Array(this.n);
    sources.forEach((s) => (marker[s] = -1));
    write(device, this.heatBuf, 0, marker);
    write(device, this.uniform, 0, new Float32Array([this.n, this.f, 0, 0]));
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(this.pipes.initSource);
    pass.setBindGroup(0, this.heatBindGroup());
    pass.dispatchWorkgroups(Math.ceil(this.n / 64));
    pass.end();
    device.queue.submit([enc.finish()]);

    const u = this.cg.solve(this.heatSystem, this.heatBuf, cgIterations);
    const enc2 = device.createCommandEncoder();
    enc2.copyBufferToBuffer(u, 0, this.heatBuf, 0, this.heatBuf.size);
    device.queue.submit([enc2.finish()]);

    const enc3 = device.createCommandEncoder();
    const pass3 = enc3.beginComputePass();
    pass3.setPipeline(this.pipes.faceGradient);
    pass3.setBindGroup(0, this.heatBindGroup());
    pass3.dispatchWorkgroups(Math.ceil(this.f / 64));
    pass3.setPipeline(this.pipes.faceDivergence);
    pass3.setBindGroup(0, this.heatBindGroup());
    pass3.dispatchWorkgroups(Math.ceil(this.f / 64));
    pass3.setPipeline(this.pipes.dequantize);
    pass3.setBindGroup(0, this.heatBindGroup());
    pass3.dispatchWorkgroups(Math.ceil(this.n / 64));
    pass3.end();
    device.queue.submit([enc3.finish()]);

    const phi = this.cg.solve(this.laplacian, this.divBuf, cgIterations);
    const enc4 = device.createCommandEncoder();
    enc4.copyBufferToBuffer(phi, 0, this.distBuf, 0, this.distBuf.size);
    device.queue.submit([enc4.finish()]);
    const phiValues = await readBufferF32(device, this.distBuf, this.n);
    const minDist = Math.min(...phiValues);
    write(device, this.uniform, 8, new Float32Array([minDist]));
    const enc5 = device.createCommandEncoder();
    const pass5 = enc5.beginComputePass();
    pass5.setPipeline(this.pipes.shift);
    pass5.setBindGroup(0, this.heatBindGroup());
    pass5.dispatchWorkgroups(Math.ceil(this.n / 64));
    pass5.end();
    device.queue.submit([enc5.finish()]);
    const distance = await readBufferF32(device, this.distBuf, this.n);
    return { distance, gpuMs: 0 };
  }

  get buffers() {
    return { distBuf: this.distBuf, n: this.n };
  }
}

export function meanEdgeLength(mesh: TriangleMesh): number {
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
