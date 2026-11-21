// GPU 几何核：法向、环境光遮蔽、标量色谱，全部走计算着色器
import { TriangleMesh } from '../core/mesh';
import { createStorageBuffer, write } from './context';
import geometryWgsl from './shaders/geometry.wgsl?raw';

const U = GPUBufferUsage;

export class GeometryKernel {
  private device: GPUDevice;
  private pipes: Record<string, GPUComputePipeline>;
  private layout: GPUBindGroupLayout;
  private uniform: GPUBuffer;
  private colorUniform: GPUBuffer;

  constructor(device: GPUDevice) {
    this.device = device;
    this.uniform = device.createBuffer({ size: 16, usage: U.UNIFORM | U.COPY_DST });
    this.colorUniform = device.createBuffer({ size: 16, usage: U.UNIFORM | U.COPY_DST });
    this.layout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        ...[1, 2, 4, 5, 7].map((b) => ({
          binding: b, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' as GPUBufferBindingType },
        })),
        ...[3, 6, 8, 9].map((b) => ({
          binding: b, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' as GPUBufferBindingType },
        })),
        { binding: 10, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      ],
    });
    const mod = device.createShaderModule({ code: geometryWgsl });
    const pl = device.createPipelineLayout({ bindGroupLayouts: [this.layout] });
    const mk = (entry: string) => device.createComputePipeline({ layout: pl, compute: { module: mod, entryPoint: entry } });
    this.pipes = {
      accNormal: mk('accumulateFaceNormals'),
      normNormal: mk('normalizeVertexNormals'),
      ao: mk('computeAO'),
      color: mk('scalarToColor'),
    };
  }

  /** 计算顶点法向与 AO，返回可直接用于渲染的缓冲数据（网格载入时调用一次） */
  async computeNormalsAndAO(mesh: TriangleMesh, aoRadius = 60, aoStrength = 1.2): Promise<{ normals: Float32Array; ao: Float32Array }> {
    const device = this.device;
    const n = mesh.vertexCount;
    const f = mesh.faceCount;
    const posBuf = createStorageBuffer(device, mesh.positions);
    const idxBuf = createStorageBuffer(device, mesh.indices);
    const normalBuf = device.createBuffer({ size: Math.ceil((n * 12) / 16) * 16, usage: U.STORAGE | U.COPY_SRC });
    const aoBuf = device.createBuffer({ size: Math.ceil((n * 4) / 16) * 16, usage: U.STORAGE | U.COPY_SRC });
    const scalarBuf = createStorageBuffer(device, new Float32Array(n));
    const colorBuf = device.createBuffer({ size: Math.ceil((n * 12) / 16) * 16, usage: U.STORAGE | U.COPY_DST });
    const accInt = device.createBuffer({ size: Math.ceil((n * 12) / 16) * 16, usage: U.STORAGE });

    // v2v CSR
    const v2v = mesh.v2v();
    const off = new Uint32Array(n + 1);
    const idxs: number[] = [];
    for (let i = 0; i < n; i++) {
      off[i] = idxs.length;
      idxs.push(...v2v[i]);
    }
    off[n] = idxs.length;
    const v2vOff = createStorageBuffer(device, off);
    const v2vIdx = createStorageBuffer(device, Uint32Array.from(idxs));

    write(device, this.uniform, 0, new Float32Array([n, f, aoRadius, aoStrength]));
    const bg = device.createBindGroup({
      layout: this.layout,
      entries: [
        { binding: 0, resource: { buffer: this.uniform } },
        { binding: 1, resource: { buffer: posBuf } },
        { binding: 2, resource: { buffer: idxBuf } },
        { binding: 3, resource: { buffer: normalBuf } },
        { binding: 4, resource: { buffer: v2vOff } },
        { binding: 5, resource: { buffer: v2vIdx } },
        { binding: 6, resource: { buffer: aoBuf } },
        { binding: 7, resource: { buffer: scalarBuf } },
        { binding: 8, resource: { buffer: colorBuf } },
        { binding: 9, resource: { buffer: accInt } },
        { binding: 10, resource: { buffer: this.colorUniform } },
      ],
    });
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setBindGroup(0, bg);
    pass.setPipeline(this.pipes.accNormal);
    pass.dispatchWorkgroups(Math.ceil(f / 64));
    pass.setPipeline(this.pipes.normNormal);
    pass.dispatchWorkgroups(Math.ceil(n / 64));
    pass.setPipeline(this.pipes.ao);
    pass.dispatchWorkgroups(Math.ceil(n / 64));
    pass.end();
    device.queue.submit([enc.finish()]);

    // 回读法向与 AO
    const normals = await this.asyncReadback(normalBuf, n * 3);
    const ao = await this.asyncReadback(aoBuf, n);
    return { normals, ao };
  }

  /** 标量场 -> 顶点色（GPU 核），mode: 0=差值 1=压力 2=厚度 */
  async scalarToColor(scalar: Float32Array, mode: 0 | 1 | 2, maxAbs: number): Promise<Float32Array> {
    const device = this.device;
    const n = scalar.length;
    const scalarBuf = createStorageBuffer(device, scalar);
    const colorBuf = device.createBuffer({ size: Math.ceil((n * 12) / 16) * 16, usage: U.STORAGE | U.COPY_SRC });
    // 占位缓冲（绑定布局要求齐全）
    const dummyPos = createStorageBuffer(device, new Float32Array(3));
    const dummyIdx = createStorageBuffer(device, new Uint32Array(1));
    const dummyNormal = device.createBuffer({ size: 16, usage: U.STORAGE });
    const dummyAo = device.createBuffer({ size: 16, usage: U.STORAGE });
    const dummyOff = createStorageBuffer(device, new Uint32Array(2));
    const dummyV2v = createStorageBuffer(device, new Uint32Array(1));
    const dummyAcc = device.createBuffer({ size: 16, usage: U.STORAGE });
    write(device, this.uniform, 0, new Float32Array([n, 0, 60, 1.2]));
    write(device, this.colorUniform, 0, new Float32Array([n, maxAbs, mode, 0]));
    const bg = device.createBindGroup({
      layout: this.layout,
      entries: [
        { binding: 0, resource: { buffer: this.uniform } },
        { binding: 1, resource: { buffer: dummyPos } },
        { binding: 2, resource: { buffer: dummyIdx } },
        { binding: 3, resource: { buffer: dummyNormal } },
        { binding: 4, resource: { buffer: dummyOff } },
        { binding: 5, resource: { buffer: dummyV2v } },
        { binding: 6, resource: { buffer: dummyAo } },
        { binding: 7, resource: { buffer: scalarBuf } },
        { binding: 8, resource: { buffer: colorBuf } },
        { binding: 9, resource: { buffer: dummyAcc } },
        { binding: 10, resource: { buffer: this.colorUniform } },
      ],
    });
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(this.pipes.color);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(Math.ceil(n / 64));
    pass.end();
    device.queue.submit([enc.finish()]);
    return this.asyncReadback(colorBuf, n * 3);
  }

  private async asyncReadback(buf: GPUBuffer, count: number): Promise<Float32Array> {
    const device = this.device;
    const rb = device.createBuffer({ size: Math.ceil((count * 4) / 16) * 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(buf, 0, rb, 0, count * 4);
    device.queue.submit([enc.finish()]);
    await rb.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(rb.getMappedRange().slice(0, count * 4));
    rb.unmap();
    rb.destroy();
    return out;
  }
}
