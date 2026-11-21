// WebGPU 设备/缓冲/管线封装 —— 计算着色器全部自写，不依赖任何图形库
export interface GpuContext {
  device: GPUDevice;
  adapter: GPUAdapter;
  format: GPUTextureFormat;
}

export async function initWebGPU(powerPreference: GPUPowerPreference = 'high-performance'): Promise<GpuContext> {
  if (!('gpu' in navigator)) {
    throw new Error('当前浏览器不支持 WebGPU，请使用 Chrome/Edge 113+ 或最新 Safari');
  }
  const adapter = await navigator.gpu.requestAdapter({ powerPreference });
  if (!adapter) throw new Error('未找到可用 GPU 适配器');
  const device = await adapter.requestDevice();
  const format = navigator.gpu.getPreferredCanvasFormat();
  device.lost.then((info) => {
    console.error('WebGPU device lost:', info);
  });
  return { device, adapter, format };
}

export function createBuffer(device: GPUDevice, data: ArrayBuffer | ArrayBufferView, usage: GPUBufferUsageFlags): GPUBuffer {
  const buf = device.createBuffer({
    size: Math.max(16, Math.ceil(data.byteLength / 16) * 16),
    usage,
    mappedAtCreation: true,
  });
  new Uint8Array(buf.getMappedRange()).set(new Uint8Array((data as ArrayBufferView).buffer ?? data, (data as ArrayBufferView).byteOffset ?? 0, data.byteLength));
  buf.unmap();
  return buf;
}

export function createStorageBuffer(device: GPUDevice, data: ArrayBufferView): GPUBuffer {
  return createBuffer(device, data, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);
}

export function createUniformBuffer(device: GPUDevice, byteLength: number): GPUBuffer {
  return device.createBuffer({
    size: Math.ceil(byteLength / 16) * 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
}

export function createReadbackBuffer(device: GPUDevice, byteLength: number): GPUBuffer {
  const size = Math.ceil(byteLength / 16) * 16;
  return device.createBuffer({ size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
}

/** 类型安全的队列写入（解决 Float32Array<ArrayBufferLike> 与 WebGPU 类型不兼容） */
export function write(
  device: GPUDevice,
  buffer: GPUBuffer,
  offset: number,
  data: ArrayBufferView,
): void {
  const copy = new Uint8Array(data.byteLength);
  copy.set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
  device.queue.writeBuffer(buffer, offset, copy.buffer);
}

export async function readBufferF32(device: GPUDevice, buf: GPUBuffer, count: number): Promise<Float32Array> {
  const rb = createReadbackBuffer(device, count * 4);
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(buf, 0, rb, 0, count * 4);
  device.queue.submit([enc.finish()]);
  await rb.mapAsync(GPUMapMode.READ);
  const out = new Float32Array(rb.getMappedRange().slice(0, count * 4));
  rb.unmap();
  rb.destroy();
  return out;
}

export async function readBufferU32(device: GPUDevice, buf: GPUBuffer, count: number): Promise<Uint32Array> {
  const rb = createReadbackBuffer(device, count * 4);
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(buf, 0, rb, 0, count * 4);
  device.queue.submit([enc.finish()]);
  await rb.mapAsync(GPUMapMode.READ);
  const out = new Uint32Array(rb.getMappedRange().slice(0, count * 4));
  rb.unmap();
  rb.destroy();
  return out;
}

/** 计算管线缓存 */
export class ComputePipelineCache {
  private device: GPUDevice;
  private modules = new Map<string, GPUShaderModule>();
  private pipelines = new Map<string, GPUComputePipeline>();

  constructor(device: GPUDevice) {
    this.device = device;
  }

  getModule(code: string, label = 'compute'): GPUShaderModule {
    let m = this.modules.get(code);
    if (!m) {
      m = this.device.createShaderModule({ code });
      this.modules.set(code, m);
    }
    return m;
  }

  getPipeline(code: string, entry: string, layout: GPUPipelineLayout, label = 'compute'): GPUComputePipeline {
    const key = `${entry}:${layout.label}:${code.length}`;
    let p = this.pipelines.get(key);
    if (!p) {
      p = this.device.createComputePipeline({
        label: `${label}.${entry}`,
        layout,
        compute: { module: this.getModule(code, label), entryPoint: entry },
      });
      this.pipelines.set(key, p);
    }
    return p;
  }
}

/** 单次计算分派封装 */
export function dispatchCompute(
  device: GPUDevice,
  pipeline: GPUComputePipeline,
  groups: GPUBindGroup[],
  workgroupCount: number,
  workgroupSize = 64,
): void {
  const enc = device.createCommandEncoder();
  const pass = enc.beginComputePass();
  pass.setPipeline(pipeline);
  groups.forEach((g, i) => pass.setBindGroup(i, g));
  pass.dispatchWorkgroups(Math.ceil(workgroupCount / workgroupSize));
  pass.end();
  device.queue.submit([enc.finish()]);
}

/** 同一 pass 内串行多个 kernel（减少 submit 开销） */
export function chainedCompute(
  device: GPUDevice,
  steps: Array<{ pipeline: GPUComputePipeline; groups: GPUBindGroup[]; count: number; size?: number }>,
): void {
  const enc = device.createCommandEncoder();
  const pass = enc.beginComputePass();
  for (const s of steps) {
    pass.setPipeline(s.pipeline);
    s.groups.forEach((g, i) => pass.setBindGroup(i, g));
    pass.dispatchWorkgroups(Math.ceil(s.count / (s.size ?? 64)));
  }
  pass.end();
  device.queue.submit([enc.finish()]);
}
