import { Mesh, Adjacency, faceCount, vertexCount, faceNormals, vertexNormals as cpuVertexNormals } from '../geometry/mesh';
import {
  FACE_NORMALS_WGSL, VERTEX_NORMALS_WGSL, BRUSH_DISPLACE_WGSL,
  OFFSET_SURFACE_WGSL, HEAT_JACOBI_WGSL,
} from './wgsl';

export interface GpuCapability {
  available: boolean;
  device?: GPUDevice;
  adapter?: GPUAdapter;
  reason?: string;
}

/**
 * WebGPU 计算运行时。检测不到适配器时所有方法回退到 src/geometry 的 CPU 实现，
 * 业务层无需感知。缓冲按需创建、按用途复用。
 */
export class GpuCompute {
  private device: GPUDevice | null = null;
  private module: GPUShaderModule | null = null;
  private pipelineCache = new Map<string, GPUComputePipeline>();

  async init(): Promise<GpuCapability> {
    if (typeof navigator === 'undefined' || !navigator.gpu) {
      return { available: false, reason: '当前浏览器不支持 WebGPU' };
    }
    try {
      const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
      if (!adapter) return { available: false, reason: '未找到可用 GPU 适配器' };
      const device = await adapter.requestDevice();
      this.device = device;
      this.module = device.createShaderModule({
        code: [
          FACE_NORMALS_WGSL, VERTEX_NORMALS_WGSL, BRUSH_DISPLACE_WGSL,
          OFFSET_SURFACE_WGSL, HEAT_JACOBI_WGSL,
        ].join('\n'),
      });
      device.onuncapturederror = (ev) => console.warn('[WebGPU] 未捕获错误', ev.error);
      return { available: true, device, adapter };
    } catch (e) {
      return { available: false, reason: `WebGPU 初始化失败：${(e as Error).message}` };
    }
  }

  get isReady(): boolean {
    return this.device !== null && this.module !== null;
  }

  private pipeline(entryPoint: string): GPUComputePipeline {
    if (!this.device || !this.module) throw new Error('WebGPU 未初始化');
    let p = this.pipelineCache.get(entryPoint);
    if (!p) {
      p = this.device.createComputePipeline({
        layout: 'auto',
        compute: { module: this.module, entryPoint },
      });
      this.pipelineCache.set(entryPoint, p);
    }
    return p;
  }

  private buffer(data: AllowSharedBufferSource, usage: GPUBufferUsageFlags): GPUBuffer {
    const device = this.device!;
    const buf = device.createBuffer({ size: Math.max(16, data.byteLength), usage });
    device.queue.writeBuffer(buf, 0, data as BufferSource);
    return buf;
  }

  private uniformBuffer(data: ArrayBuffer): GPUBuffer {
    const device = this.device!;
    const size = Math.max(16, Math.ceil(data.byteLength / 16) * 16);
    const buf = device.createBuffer({ size, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(buf, 0, data);
    return buf;
  }

  private async readToFloat32(buf: GPUBuffer, count: number): Promise<Float32Array> {
    const device = this.device!;
    const staging = device.createBuffer({
      size: buf.size,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(buf, 0, staging, 0, buf.size);
    device.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(staging.getMappedRange().slice(0, count * 4));
    staging.unmap();
    staging.destroy();
    return out;
  }

  private destroy(bufs: GPUBuffer[]): void {
    for (const b of bufs) b.destroy();
  }

  /** GPU 面积加权顶点法线；失败回退 CPU */
  async computeVertexNormals(mesh: Mesh, adj: Adjacency): Promise<Float32Array> {
    if (!this.isReady) return cpuVertexNormals(mesh);
    try {
      const device = this.device!;
      const V = vertexCount(mesh);
      const F = faceCount(mesh);
      const positions = this.buffer(mesh.positions, GPUBufferUsage.STORAGE);
      const indices = this.buffer(mesh.indices, GPUBufferUsage.STORAGE);
      const faceN = device.createBuffer({
        size: Math.max(16, F * 3 * 4),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      });
      const outN = device.createBuffer({
        size: Math.max(16, V * 3 * 4),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      });

      // pass 1 面法线
      const meta1 = this.uniformBuffer(new Uint32Array([F, 0, 0, 0]).buffer);
      const p1 = this.pipeline('main');
      const bind1 = device.createBindGroup({
        layout: p1.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: meta1 } },
          { binding: 1, resource: { buffer: positions } },
          { binding: 2, resource: { buffer: indices } },
          { binding: 3, resource: { buffer: faceN } },
        ],
      });
      // pass 2 顶点法线
      const faceOffset = this.buffer(adj.faceOffset, GPUBufferUsage.STORAGE);
      const vertexFaces = this.buffer(adj.vertexFaces, GPUBufferUsage.STORAGE);
      const meta2 = this.uniformBuffer(new Uint32Array([V, 0, 0, 0]).buffer);
      const p2 = this.pipeline('main');
      const bind2 = device.createBindGroup({
        layout: p2.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: meta2 } },
          { binding: 1, resource: { buffer: faceN } },
          { binding: 2, resource: { buffer: faceOffset } },
          { binding: 3, resource: { buffer: vertexFaces } },
          { binding: 4, resource: { buffer: outN } },
        ],
      });

      const enc = device.createCommandEncoder();
      const pass = enc.beginComputePass();
      pass.setPipeline(p1);
      pass.setBindGroup(0, bind1);
      pass.dispatchWorkgroups(Math.ceil(F / 64));
      pass.setPipeline(p2);
      pass.setBindGroup(0, bind2);
      pass.dispatchWorkgroups(Math.ceil(V / 256));
      pass.end();
      device.queue.submit([enc.finish()]);

      const result = await this.readToFloat32(outN, V * 3);
      this.destroy([positions, indices, faceN, outN, meta1, faceOffset, vertexFaces, meta2]);
      return result;
    } catch (e) {
      console.warn('[WebGPU] 法线计算回退 CPU：', e);
      return cpuVertexNormals(mesh);
    }
  }

  /** GPU 笔刷位移（就地改 positions），返回带符号位移场；失败回退由调用方处理 */
  async brushDisplace(
    positions: Float32Array,
    normals: Float32Array,
    distances: Float32Array,
    sign: number,
    radiusMm: number,
    depthMm: number,
    centerVertex: number,
  ): Promise<{ positions: Float32Array; displacement: Float32Array } | null> {
    if (!this.isReady) return null;
    try {
      const device = this.device!;
      const V = distances.length;
      const pos = this.buffer(positions, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
      const nrm = this.buffer(normals, GPUBufferUsage.STORAGE);
      const dist = this.buffer(distances, GPUBufferUsage.STORAGE);
      const disp = device.createBuffer({
        size: Math.max(16, V * 4),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      });
      // uniform: u32 center, f32 sign, f32 radius, f32 depth
      const u = new ArrayBuffer(16);
      new Uint32Array(u, 0, 1)[0] = centerVertex;
      new Float32Array(u, 4, 3).set([sign, radiusMm, depthMm]);
      const uniform = this.uniformBuffer(u);
      const pipe = this.pipeline('main');
      const bind = device.createBindGroup({
        layout: pipe.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: uniform } },
          { binding: 1, resource: { buffer: dist } },
          { binding: 2, resource: { buffer: nrm } },
          { binding: 3, resource: { buffer: pos } },
          { binding: 4, resource: { buffer: disp } },
        ],
      });
      const enc = device.createCommandEncoder();
      const pass = enc.beginComputePass();
      pass.setPipeline(pipe);
      pass.setBindGroup(0, bind);
      pass.dispatchWorkgroups(Math.ceil(V / 256));
      pass.end();
      device.queue.submit([enc.finish()]);
      const outPos = await this.readToFloat32(pos, positions.length);
      const outDisp = await this.readToFloat32(disp, V);
      this.destroy([pos, nrm, dist, disp, uniform]);
      return { positions: outPos, displacement: outDisp };
    } catch (e) {
      console.warn('[WebGPU] 笔刷位移回退 CPU：', e);
      return null;
    }
  }

  /** GPU 法向等距偏移生成外壁；失败返回 null 由调用方走 CPU */
  async offsetSurface(
    innerPos: Float32Array,
    normals: Float32Array,
    thickness: Float32Array,
  ): Promise<Float32Array | null> {
    if (!this.isReady) return null;
    try {
      const device = this.device!;
      const V = thickness.length;
      const inner = this.buffer(innerPos, GPUBufferUsage.STORAGE);
      const nrm = this.buffer(normals, GPUBufferUsage.STORAGE);
      const thk = this.buffer(thickness, GPUBufferUsage.STORAGE);
      const outer = device.createBuffer({
        size: Math.max(16, innerPos.byteLength),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      });
      const meta = this.uniformBuffer(new Uint32Array([V, 0, 0, 0]).buffer);
      const pipe = this.pipeline('main');
      const bind = device.createBindGroup({
        layout: pipe.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: meta } },
          { binding: 1, resource: { buffer: inner } },
          { binding: 2, resource: { buffer: nrm } },
          { binding: 3, resource: { buffer: thk } },
          { binding: 4, resource: { buffer: outer } },
        ],
      });
      const enc = device.createCommandEncoder();
      const pass = enc.beginComputePass();
      pass.setPipeline(pipe);
      pass.setBindGroup(0, bind);
      pass.dispatchWorkgroups(Math.ceil(V / 256));
      pass.end();
      device.queue.submit([enc.finish()]);
      const out = await this.readToFloat32(outer, innerPos.length);
      this.destroy([inner, nrm, thk, outer, meta]);
      return out;
    } catch (e) {
      console.warn('[WebGPU] 等距偏移回退 CPU：', e);
      return null;
    }
  }

  /**
   * GPU Jacobi 热扩散（热法第一步），迭代到收敛或上限。
   * 返回热场 u；不收敛/不支持时返回 null。
   */
  async heatJacobi(
    V: number,
    areas: Float32Array,
    adj: Adjacency,
    timeStep: number,
    sourceVertex: number,
    iterations = 120,
    tolerance = 1e-7,
  ): Promise<Float32Array | null> {
    if (!this.isReady) return null;
    try {
      const device = this.device!;
      const degree = new Uint32Array(V);
      for (let i = 0; i < V; i++) degree[i] = adj.offset[i + 1] - adj.offset[i];
      const rhs = new Float32Array(V);
      rhs[sourceVertex] = 1;
      let cur = this.buffer(new Float32Array(V), GPUBufferUsage.STORAGE);
      const areasB = this.buffer(areas, GPUBufferUsage.STORAGE);
      const degreeB = this.buffer(degree, GPUBufferUsage.STORAGE);
      const offsetB = this.buffer(adj.offset, GPUBufferUsage.STORAGE);
      const nbrB = this.buffer(adj.neighbors, GPUBufferUsage.STORAGE);
      const rhsB = this.buffer(rhs, GPUBufferUsage.STORAGE);
      const meta = this.uniformBuffer(
        (() => { const b = new ArrayBuffer(16); new Uint32Array(b, 0, 1)[0] = V; new Float32Array(b, 4, 1)[0] = timeStep; return b; })(),
      );
      const pipe = this.pipeline('main');
      let result: Float32Array | null = null;
      for (let it = 0; it < iterations; it++) {
        const nxt = device.createBuffer({
          size: Math.max(16, V * 4),
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });
        const bind = device.createBindGroup({
          layout: pipe.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: meta } },
            { binding: 1, resource: { buffer: cur } },
            { binding: 2, resource: { buffer: nxt } },
            { binding: 3, resource: { buffer: areasB } },
            { binding: 4, resource: { buffer: degreeB } },
            { binding: 5, resource: { buffer: offsetB } },
            { binding: 6, resource: { buffer: nbrB } },
            { binding: 7, resource: { buffer: rhsB } },
          ],
        });
        const enc = device.createCommandEncoder();
        const pass = enc.beginComputePass();
        pass.setPipeline(pipe);
        pass.setBindGroup(0, bind);
        pass.dispatchWorkgroups(Math.ceil(V / 256));
        pass.end();
        device.queue.submit([enc.finish()]);
        if (it % 8 === 7 || it === iterations - 1) {
          const prev = await this.readToFloat32(cur, V);
          result = await this.readToFloat32(nxt, V);
          let diff = 0;
          for (let i = 0; i < V; i++) diff = Math.max(diff, Math.abs(result[i] - prev[i]));
          if (diff < tolerance) { cur.destroy(); nxt.destroy(); break; }
        }
        cur.destroy();
        cur = nxt;
      }
      this.destroy([areasB, degreeB, offsetB, nbrB, rhsB, meta]);
      if (result) cur.destroy();
      return result;
    } catch (e) {
      console.warn('[WebGPU] 热扩散回退 CPU：', e);
      return null;
    }
  }
}

/** 全局单例，应用启动时 init 一次 */
export const gpuCompute = new GpuCompute();
export { faceNormals };
