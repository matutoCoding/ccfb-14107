// WebGPU 渲染器：自写渲染管线，支持实体/PETG 半透明/色谱模式、AO、线框、标记点、射线拾取
import { Mat4, Vec3, lookAt4, perspective4, multiply4, sub, normalize, cross } from '../core/math';
import { TriangleMesh, rayTriangle } from '../core/mesh';
import { write } from './context';
import { GeometryKernel } from './geometry-kernel';
import renderWgsl from './shaders/render.wgsl?raw';

export interface RenderMaterial {
  baseColor: [number, number, number];
  opacity: number;
  /** 0=实体材质 1=顶点色谱 */
  mode: number;
  aoStrength: number;
  specular: number;
}

export interface RenderMesh {
  id: string;
  positions: GPUBuffer;
  normals: GPUBuffer;
  colors: GPUBuffer;
  ao: GPUBuffer;
  indexBuffer: GPUBuffer;
  indexCount: number;
  material: RenderMaterial;
  visible: boolean;
}

export interface OrbitState {
  yaw: number;
  pitch: number;
  distance: number;
  target: Vec3;
}

const U = GPUBufferUsage;

export class Renderer {
  device: GPUDevice;
  canvas: HTMLCanvasElement;
  context: GPUCanvasContext;
  format: GPUTextureFormat;
  private cameraUniform: GPUBuffer;
  private renderPipe: GPURenderPipeline;
  private linePipe: GPURenderPipeline;
  private meshes = new Map<string, RenderMesh>();
  private depthTexture: GPUTexture | null = null;
  orbit: OrbitState = { yaw: 0.6, pitch: 0.25, distance: 900, target: [0, 260, 0] };
  private pickMesh: TriangleMesh | null = null;
  private geometryKernel: GeometryKernel | null = null;

  constructor(device: GPUDevice, canvas: HTMLCanvasElement) {
    this.device = device;
    this.canvas = canvas;
    this.format = navigator.gpu.getPreferredCanvasFormat();
    this.context = canvas.getContext('webgpu')!;
    this.context.configure({ device, format: this.format, alphaMode: 'opaque' });
    this.cameraUniform = device.createBuffer({ size: 64, usage: U.UNIFORM | U.COPY_DST });

    const cameraLayout = device.createBindGroupLayout({
      entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } }],
    });
    const objectLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        ...[1, 2, 3, 4, 6].map((b) => ({
          binding: b,
          visibility: GPUShaderStage.VERTEX,
          buffer: { type: 'read-only-storage' as GPUBufferBindingType },
        })),
        { binding: 5, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      ],
    });
    const mod = device.createShaderModule({ code: renderWgsl });
    const layout = device.createPipelineLayout({ bindGroupLayouts: [cameraLayout, objectLayout] });
    this.renderPipe = device.createRenderPipeline({
      layout,
      vertex: { module: mod, entryPoint: 'vsMain', buffers: [] },
      fragment: {
        module: mod,
        entryPoint: 'fsMain',
        targets: [{
          format: this.format,
          blend: {
            color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
            alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
          },
        }],
      },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      depthStencil: {
        depthWriteEnabled: true,
        depthCompare: 'less',
        format: 'depth24plus',
      },
    });
    this.linePipe = device.createRenderPipeline({
      layout,
      vertex: { module: mod, entryPoint: 'vsMain', buffers: [] },
      fragment: { module: mod, entryPoint: 'fsLine', targets: [{ format: this.format }] },
      primitive: { topology: 'line-list', cullMode: 'none' },
      depthStencil: { depthWriteEnabled: true, depthCompare: 'less', format: 'depth24plus' },
    });

    this.attachOrbitControls();
  }

  setPickMesh(mesh: TriangleMesh | null) {
    this.pickMesh = mesh;
  }

  private attachOrbitControls() {
    let dragging = false;
    let lastX = 0;
    let lastY = 0;
    const canvas = this.canvas;
    canvas.addEventListener('pointerdown', (e) => {
      dragging = true;
      lastX = e.clientX;
      lastY = e.clientY;
      canvas.setPointerCapture(e.pointerId);
    });
    canvas.addEventListener('pointermove', (e) => {
      if (!dragging) return;
      this.orbit.yaw -= (e.clientX - lastX) * 0.008;
      this.orbit.pitch = Math.max(-1.3, Math.min(1.3, this.orbit.pitch + (e.clientY - lastY) * 0.008));
      lastX = e.clientX;
      lastY = e.clientY;
    });
    canvas.addEventListener('pointerup', () => (dragging = false));
    canvas.addEventListener('wheel', (e) => {
      e.preventDefault();
      this.orbit.distance = Math.max(150, Math.min(2500, this.orbit.distance * (1 + Math.sign(e.deltaY) * 0.08)));
    }, { passive: false });
  }

  /** 屏幕坐标拾取，返回网格面与重心坐标 */
  pick(clientX: number, clientY: number): { face: number; point: Vec3; vertex: number } | null {
    if (!this.pickMesh) return null;
    const rect = this.canvas.getBoundingClientRect();
    const nx = ((clientX - rect.left) / rect.width) * 2 - 1;
    const ny = 1 - ((clientY - rect.top) / rect.height) * 2;
    const { viewProj, eye } = this.cameraMatrices();
    const inv = invertMat(viewProj);
    const near = unproject(inv, [nx, ny, 0]);
    const far = unproject(inv, [nx, ny, 1]);
    const origin = eye;
    const dir = normalize(sub(far, near));
    let best: { t: number; face: number } | null = null;
    for (let f = 0; f < this.pickMesh.faceCount; f++) {
      const hit = rayTriangle(this.pickMesh, f, origin, dir);
      if (hit && (!best || hit.t < best.t)) best = { t: hit.t, face: f };
    }
    if (!best) return null;
    const [a, b, c] = this.pickMesh.face(best.face);
    const pa = this.pickMesh.vertex(a);
    // 取离射线命中点最近的顶点
    const hitPoint: Vec3 = [
      origin[0] + dir[0] * best.t,
      origin[1] + dir[1] * best.t,
      origin[2] + dir[2] * best.t,
    ];
    const verts = [a, b, c];
    let nearest = verts[0];
    let nd = Infinity;
    for (const v of verts) {
      const p = this.pickMesh.vertex(v);
      const d = (p[0] - hitPoint[0]) ** 2 + (p[1] - hitPoint[1]) ** 2 + (p[2] - hitPoint[2]) ** 2;
      if (d < nd) {
        nd = d;
        nearest = v;
      }
    }
    void pa;
    return { face: best.face, point: hitPoint, vertex: nearest };
  }

  private cameraMatrices(): { viewProj: Mat4; eye: Vec3 } {
    const aspect = this.canvas.clientWidth / Math.max(1, this.canvas.clientHeight);
    const proj = perspective4((50 * Math.PI) / 180, aspect, 5, 6000);
    const { yaw, pitch, distance, target } = this.orbit;
    const eye: Vec3 = [
      target[0] + distance * Math.cos(pitch) * Math.sin(yaw),
      target[1] + distance * Math.sin(pitch),
      target[2] + distance * Math.cos(pitch) * Math.cos(yaw),
    ];
    const view = lookAt4(eye, target, [0, 1, 0]);
    return { viewProj: multiply4(proj, view), eye };
  }

  addMesh(id: string, mesh: TriangleMesh, material: Partial<RenderMaterial> = {}): RenderMesh {
    const n = mesh.vertexCount;
    const normals = mesh.vertexNormals();
    const colors = mesh.colors ?? new Float32Array(n * 3).fill(0.7);
    const ao = new Float32Array(n).fill(1);
    const positions = this.device.createBuffer({ size: Math.ceil((n * 3 * 4) / 16) * 16, usage: U.STORAGE | U.COPY_DST });
    write(this.device, positions, 0, mesh.positions);
    const normalBuf = this.device.createBuffer({ size: positions.size, usage: U.STORAGE | U.COPY_DST });
    write(this.device, normalBuf, 0, normals);
    const colorBuf = this.device.createBuffer({ size: positions.size, usage: U.STORAGE | U.COPY_DST });
    write(this.device, colorBuf, 0, colors);
    const aoBuf = this.device.createBuffer({ size: Math.ceil((n * 4) / 16) * 16, usage: U.STORAGE | U.COPY_DST });
    write(this.device, aoBuf, 0, ao);
    const indexBuffer = this.device.createBuffer({
      size: Math.ceil((mesh.indices.byteLength) / 16) * 16,
      usage: U.STORAGE | U.COPY_DST,
    });
    write(this.device, indexBuffer, 0, mesh.indices);
    const rm: RenderMesh = {
      id,
      positions,
      normals: normalBuf,
      colors: colorBuf,
      ao: aoBuf,
      indexBuffer,
      indexCount: mesh.indices.length,
      material: {
        baseColor: [0.82, 0.78, 0.72],
        opacity: 1,
        mode: 0,
        aoStrength: 0.7,
        specular: 0.35,
        ...material,
      },
      visible: true,
    };
    this.meshes.set(id, rm);
    // GPU 法向 + AO 异步增强（不阻塞首帧）
    void this.enrichWithGPU(id, mesh);
    return rm;
  }

  private async enrichWithGPU(id: string, mesh: TriangleMesh) {
    try {
      if (!this.geometryKernel) this.geometryKernel = new GeometryKernel(this.device);
      const { normals, ao } = await this.geometryKernel.computeNormalsAndAO(mesh);
      const rm = this.meshes.get(id);
      if (!rm) return;
      write(this.device, rm.normals, 0, normals);
      write(this.device, rm.ao, 0, ao);
    } catch (e) {
      // GPU 核失败时保留 CPU 法向，AO 用 1（无遮挡），渲染不中断
      console.warn('GPU 几何核回退 CPU：', e);
    }
  }

  /** 用 GPU 色谱核替换顶点色（mode 0=差值 1=压力 2=厚度） */
  async applyScalarColor(id: string, scalar: Float32Array, mode: 0 | 1 | 2, maxAbs: number) {
    const rm = this.meshes.get(id);
    if (!rm) return;
    if (!this.geometryKernel) this.geometryKernel = new GeometryKernel(this.device);
    const colors = await this.geometryKernel.scalarToColor(scalar, mode, maxAbs);
    write(this.device, rm.colors, 0, colors);
    rm.material = { ...rm.material, mode: 1 };
  }

  updatePositions(id: string, positions: Float32Array, normals?: Float32Array) {
    const rm = this.meshes.get(id);
    if (!rm) return;
    write(this.device, rm.positions, 0, positions);
    if (normals) write(this.device, rm.normals, 0, normals);
  }

  updateColors(id: string, colors: Float32Array) {
    const rm = this.meshes.get(id);
    if (rm) write(this.device, rm.colors, 0, colors);
  }

  updateAO(id: string, ao: Float32Array) {
    const rm = this.meshes.get(id);
    if (rm) write(this.device, rm.ao, 0, ao);
  }

  setMaterial(id: string, patch: Partial<RenderMaterial>) {
    const rm = this.meshes.get(id);
    if (rm) rm.material = { ...rm.material, ...patch };
  }

  setVisible(id: string, visible: boolean) {
    const rm = this.meshes.get(id);
    if (rm) rm.visible = visible;
  }

  removeMesh(id: string) {
    const rm = this.meshes.get(id);
    if (rm) {
      [rm.positions, rm.normals, rm.colors, rm.ao, rm.indexBuffer].forEach((b) => b.destroy());
      this.meshes.delete(id);
    }
  }

  /** 清空全部网格（切换页面/工作对象时调用，释放 GPU 缓冲） */
  clearMeshes() {
    for (const id of Array.from(this.meshes.keys())) this.removeMesh(id);
    this.setPickMesh(null);
  }

  getMesh(id: string) {
    return this.meshes.get(id);
  }

  render() {
    const canvas = this.canvas;
    const w = canvas.clientWidth * (window.devicePixelRatio || 1);
    const h = canvas.clientHeight * (window.devicePixelRatio || 1);
    if (canvas.width !== Math.floor(w) || canvas.height !== Math.floor(h)) {
      canvas.width = Math.floor(w);
      canvas.height = Math.floor(h);
    }
    if (!this.depthTexture || this.depthTexture.width !== canvas.width) {
      this.depthTexture?.destroy();
      this.depthTexture = this.device.createTexture({
        size: [canvas.width, canvas.height],
        format: 'depth24plus',
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
      });
    }
    const { viewProj, eye } = this.cameraMatrices();
    write(this.device, this.cameraUniform, 0, viewProj);
    write(this.device, this.cameraUniform, 16, new Float32Array([...eye, 1]));

    const enc = this.device.createCommandEncoder();
    const pass = enc.beginRenderPass({
      colorAttachments: [{
        view: this.context.getCurrentTexture().createView(),
        clearValue: { r: 0.09, g: 0.1, b: 0.12, a: 1 },
        loadOp: 'clear',
        storeOp: 'store',
      }],
      depthStencilAttachment: {
        view: this.depthTexture.createView(),
        depthClearValue: 1,
        depthLoadOp: 'clear',
        depthStoreOp: 'store',
      },
    });

    const cameraBG = this.device.createBindGroup({
      layout: this.renderPipe.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: this.cameraUniform } }],
    });

    // 不透明先画，半透明 PETG 后画
    const ordered = Array.from(this.meshes.values()).filter((m) => m.visible);
    ordered.sort((a, b) => a.material.opacity - b.material.opacity).reverse();
    for (const rm of ordered) {
      const matBuf = this.device.createBuffer({ size: 32, usage: U.UNIFORM | U.COPY_DST });
      write(this.device, 
        matBuf,
        0,
        new Float32Array([
          ...rm.material.baseColor,
          rm.material.opacity,
          rm.material.mode,
          rm.material.aoStrength,
          rm.material.specular,
          0,
        ]),
      );
      const lineMatBuf = this.device.createBuffer({ size: 32, usage: U.UNIFORM | U.COPY_DST });
      write(this.device, lineMatBuf, 0, new Float32Array([0.95, 0.4, 0.2, 1, 0, 0, 0, 0]));
      const objectBG = this.device.createBindGroup({
        layout: this.renderPipe.getBindGroupLayout(1),
        entries: [
          { binding: 0, resource: { buffer: matBuf } },
          { binding: 1, resource: { buffer: rm.positions } },
          { binding: 2, resource: { buffer: rm.normals } },
          { binding: 3, resource: { buffer: rm.colors } },
          { binding: 4, resource: { buffer: rm.ao } },
          { binding: 5, resource: { buffer: lineMatBuf } },
          { binding: 6, resource: { buffer: rm.indexBuffer } },
        ],
      });
      pass.setPipeline(this.renderPipe);
      pass.setBindGroup(0, cameraBG);
      pass.setBindGroup(1, objectBG);
      pass.draw(rm.indexCount);
    }
    pass.end();
    this.device.queue.submit([enc.finish()]);
  }
}

// —— 矩阵工具（拾取反投影用）——
function unproject(inv: Mat4, ndc: [number, number, number]): Vec3 {
  const x = inv[0] * ndc[0] + inv[4] * ndc[1] + inv[8] * ndc[2] + inv[12];
  const y = inv[1] * ndc[0] + inv[5] * ndc[1] + inv[9] * ndc[2] + inv[13];
  const z = inv[2] * ndc[0] + inv[6] * ndc[1] + inv[10] * ndc[2] + inv[14];
  const w = inv[3] * ndc[0] + inv[7] * ndc[1] + inv[11] * ndc[2] + inv[15];
  return [x / w, y / w, z / w];
}

function invertMat(a: Mat4): Mat4 {
  const out = new Float32Array(16);
  const a00 = a[0], a01 = a[1], a02 = a[2], a03 = a[3];
  const a10 = a[4], a11 = a[5], a12 = a[6], a13 = a[7];
  const a20 = a[8], a21 = a[9], a22 = a[10], a23 = a[11];
  const a30 = a[12], a31 = a[13], a32 = a[14], a33 = a[15];
  const b00 = a00 * a11 - a01 * a10;
  const b01 = a00 * a12 - a02 * a10;
  const b02 = a00 * a13 - a03 * a10;
  const b03 = a01 * a12 - a02 * a11;
  const b04 = a01 * a13 - a03 * a11;
  const b05 = a02 * a13 - a03 * a12;
  const b06 = a20 * a31 - a21 * a30;
  const b07 = a20 * a32 - a22 * a30;
  const b08 = a20 * a33 - a23 * a30;
  const b09 = a21 * a32 - a22 * a31;
  const b10 = a21 * a33 - a23 * a31;
  const b11 = a22 * a33 - a23 * a32;
  let det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
  if (Math.abs(det) < 1e-12) return new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
  det = 1 / det;
  out[0] = (a11 * b11 - a12 * b10 + a13 * b09) * det;
  out[1] = (a02 * b10 - a01 * b11 - a03 * b09) * det;
  out[2] = (a31 * b05 - a32 * b04 + a33 * b03) * det;
  out[3] = (a22 * b04 - a21 * b05 - a23 * b03) * det;
  out[4] = (a12 * b08 - a10 * b11 - a13 * b07) * det;
  out[5] = (a00 * b11 - a02 * b08 + a03 * b07) * det;
  out[6] = (a32 * b02 - a30 * b05 - a33 * b01) * det;
  out[7] = (a20 * b05 - a22 * b02 + a23 * b01) * det;
  out[8] = (a10 * b10 - a11 * b08 + a13 * b06) * det;
  out[9] = (a01 * b08 - a00 * b10 - a03 * b06) * det;
  out[10] = (a30 * b04 - a31 * b02 + a33 * b00) * det;
  out[11] = (a21 * b02 - a20 * b04 - a23 * b00) * det;
  out[12] = (a11 * b07 - a10 * b09 - a12 * b06) * det;
  out[13] = (a00 * b09 - a01 * b07 + a02 * b06) * det;
  out[14] = (a31 * b01 - a30 * b03 - a32 * b00) * det;
  out[15] = (a20 * b03 - a21 * b01 + a22 * b00) * det;
  return out;
}

void cross;
