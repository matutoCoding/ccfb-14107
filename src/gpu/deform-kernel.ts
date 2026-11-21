// GPU 修型变形核：测地高斯衰减沿法向位移，等距偏移
import { TriangleMesh } from '../core/mesh';
import { createStorageBuffer, write } from './context';
import deformWgsl from './shaders/deform.wgsl?raw';

const U = GPUBufferUsage;

// WGSL Brush 结构：u32 center, u32 kind, f32 sigma, f32 depth, f32x2 pad = 24 字节
const BRUSH_STRIDE = 24;

export interface GpuBrush {
  centerVertex: number;
  kind: 0 | 1; // 0=压垫 1=释放
  sigma: number;
  depthMm: number;
}

export class DeformKernel {
  private device: GPUDevice;
  private brushPipe: GPUComputePipeline;
  private allPipe: GPUComputePipeline;
  private offsetPipe: GPUComputePipeline;
  private layout: GPUBindGroupLayout;
  private offsetLayout: GPUBindGroupLayout;
  private deformUniform: GPUBuffer;
  private offsetUniform: GPUBuffer;

  constructor(device: GPUDevice) {
    this.device = device;
    this.deformUniform = device.createBuffer({ size: 16, usage: U.UNIFORM | U.COPY_DST });
    this.offsetUniform = device.createBuffer({ size: 16, usage: U.UNIFORM | U.COPY_DST });
    const mod = device.createShaderModule({ code: deformWgsl });
    // applyAllBrushes / applyBrush 绑定 0..6：1-4 只读，5/6 读写
    this.layout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        ...[1, 2, 3, 4].map((b) => ({
          binding: b,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: 'read-only-storage' as GPUBufferBindingType },
        })),
        ...[5, 6].map((b) => ({
          binding: b,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: 'storage' as GPUBufferBindingType },
        })),
      ],
    });
    // offsetSurface 绑定 0,1,2,5,7,8
    this.offsetLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      ],
    });
    const plAll = device.createPipelineLayout({ bindGroupLayouts: [this.layout] });
    const plOff = device.createPipelineLayout({ bindGroupLayouts: [this.offsetLayout] });
    this.allPipe = device.createComputePipeline({ layout: plAll, compute: { module: mod, entryPoint: 'applyAllBrushes' } });
    this.brushPipe = device.createComputePipeline({ layout: plAll, compute: { module: mod, entryPoint: 'applyBrush' } });
    this.offsetPipe = device.createComputePipeline({ layout: plOff, compute: { module: mod, entryPoint: 'offsetSurface' } });
  }

  /**
   * 多笔刷一次性叠加变形
   * @param geoDistances 每个笔刷一段长度 n 的测地距离场，平铺
   */
  applyAllBrushes(
    mesh: TriangleMesh,
    normals: Float32Array,
    brushes: GpuBrush[],
    geoDistances: Float32Array,
  ): GPUBuffer {
    const device = this.device;
    const n = mesh.vertexCount;
    const posBuf = createStorageBuffer(device, mesh.positions);
    const nrmBuf = createStorageBuffer(device, normals);
    const distBuf = createStorageBuffer(device, geoDistances);
    const outBuf = device.createBuffer({ size: Math.ceil((n * 12) / 16) * 16, usage: U.STORAGE | U.COPY_SRC | U.COPY_DST });
    const weightBuf = device.createBuffer({ size: Math.ceil((n * 4) / 16) * 16, usage: U.STORAGE });
    const brushData = new Float32Array(Math.max(1, brushes.length) * (BRUSH_STRIDE / 4));
    brushes.forEach((b, i) => {
      const view = new DataView(brushData.buffer, i * BRUSH_STRIDE, BRUSH_STRIDE);
      view.setUint32(0, b.centerVertex, true);
      view.setUint32(4, b.kind, true);
      view.setFloat32(8, b.sigma, true);
      view.setFloat32(12, b.depthMm, true);
    });
    const brushBuf = createStorageBuffer(device, brushData);
    write(device, this.deformUniform, 0, new Float32Array([n, brushes.length, 0, 0]));
    const bg = device.createBindGroup({
      layout: this.layout,
      entries: [
        { binding: 0, resource: { buffer: this.deformUniform } },
        { binding: 1, resource: { buffer: posBuf } },
        { binding: 2, resource: { buffer: nrmBuf } },
        { binding: 3, resource: { buffer: distBuf } },
        { binding: 4, resource: { buffer: brushBuf } },
        { binding: 5, resource: { buffer: outBuf } },
        { binding: 6, resource: { buffer: weightBuf } },
      ],
    });
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(this.allPipe);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(Math.ceil(n / 64));
    pass.end();
    device.queue.submit([enc.finish()]);
    return outBuf;
  }

  /** 等距偏移：newPos = pos + direction * normal * thickness(v) */
  get gpuDevice(): GPUDevice {
    return this.device;
  }

  offsetSurface(
    mesh: TriangleMesh,
    normals: Float32Array,
    thickness: Float32Array,
    direction: 1 | -1,
  ): GPUBuffer {
    const device = this.device;
    const n = mesh.vertexCount;
    const posBuf = createStorageBuffer(device, mesh.positions);
    const nrmBuf = createStorageBuffer(device, normals);
    const thickBuf = createStorageBuffer(device, thickness);
    const outBuf = device.createBuffer({ size: Math.ceil((n * 12) / 16) * 16, usage: U.STORAGE | U.COPY_SRC });
    write(device, this.deformUniform, 0, new Float32Array([n, 0, 0, 0]));
    write(device, this.offsetUniform, 0, new Float32Array([n, direction, 0, 0]));
    const bg = device.createBindGroup({
      layout: this.offsetLayout,
      entries: [
        { binding: 0, resource: { buffer: this.deformUniform } },
        { binding: 1, resource: { buffer: posBuf } },
        { binding: 2, resource: { buffer: nrmBuf } },
        { binding: 5, resource: { buffer: outBuf } },
        { binding: 7, resource: { buffer: this.offsetUniform } },
        { binding: 8, resource: { buffer: thickBuf } },
      ],
    });
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(this.offsetPipe);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(Math.ceil(n / 64));
    pass.end();
    device.queue.submit([enc.finish()]);
    return outBuf;
  }
}
