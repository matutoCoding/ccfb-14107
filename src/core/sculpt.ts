// 修型引擎：压垫/释放区测地笔刷，对侧自动扩张空腔
import { TriangleMesh } from './mesh';
import { HeatGeodesic } from '../gpu/geodesic';
import { DeformKernel, GpuBrush } from '../gpu/deform-kernel';
import { readBufferF32 } from '../gpu/context';
import { Vec3, gaussian, normalize, sub } from './math';

export type { BrushKind, SculptDab, SculptPlan, SculptResult, CrossSection } from './sculpt-types';
export { defaultDab, extractCrossSections } from './sculpt-types';
import type { SculptDab, SculptResult } from './sculpt-types';

export class SculptEngine {
  private mesh: TriangleMesh;
  private geodesic: HeatGeodesic;
  private normals: Float32Array;
  private deform: DeformKernel | null;

  constructor(mesh: TriangleMesh, geodesic: HeatGeodesic, device?: GPUDevice) {
    this.mesh = mesh;
    this.geodesic = geodesic;
    this.normals = mesh.vertexNormals();
    this.deform = device ? new DeformKernel(device) : null;
  }

  /**
   * 应用整组修型 dab（在基线上一次性叠加，保证可重复、可撤销）
   * 测地距离场由 GPU 热法求解，顶点位移优先走 GPU 变形核（20 万面 <200ms），失败回退 CPU。
   */
  async applyPlan(basePositions: Float32Array, dabs: SculptDab[]): Promise<SculptResult> {
    const n = this.mesh.vertexCount;
    const fields: Float32Array[] = [];
    // 展开为 GPU 笔刷（对侧空腔转成独立的释放类笔刷）
    const gpuBrushes: GpuBrush[] = [];
    const flatFields: Float32Array[] = [];

    for (const dab of dabs) {
      const { distance } = await this.geodesic.solve(dab.centerVertex);
      fields.push(distance);
      flatFields.push(distance);
      gpuBrushes.push({
        centerVertex: dab.centerVertex,
        kind: dab.kind === 'pressure' ? 0 : 1,
        sigma: dab.sigmaMm,
        depthMm: dab.depthMm,
      });
      if (dab.kind === 'pressure' && dab.contralateral?.enabled) {
        const mirrorVertex = this.findContralateralVertex(dab.centerVertex);
        const mirrorField = await this.geodesic.solve(mirrorVertex);
        fields.push(mirrorField.distance);
        flatFields.push(mirrorField.distance);
        gpuBrushes.push({
          centerVertex: mirrorVertex,
          kind: 1, // 释放（向外 +n）
          sigma: dab.sigmaMm * (dab.contralateral.sigmaScale || 1.2),
          depthMm: dab.contralateral.depthMm,
        });
      }
    }

    if (this.deform && gpuBrushes.length > 0) {
      try {
        const geoDistances = new Float32Array(n * gpuBrushes.length);
        flatFields.forEach((f, i) => geoDistances.set(f, i * n));
        const baseMesh = new TriangleMesh(new Float32Array(basePositions), new Uint32Array(this.mesh.indices));
        const outBuf = this.deform.applyAllBrushes(baseMesh, this.normals, gpuBrushes, geoDistances);
        const positions = await readBufferF32(this.deformDevice(), outBuf, n * 3);
        return { positions, fields, gpuMs: 0 };
      } catch (e) {
        console.warn('GPU 变形回退 CPU：', e);
      }
    }
    return this.applyPlanCPU(basePositions, dabs, fields);
  }

  private deformDevice(): GPUDevice {
    return this.deform!.gpuDevice;
  }

  /** CPU 参考实现（GPU 不可用时的回退，也是单元测试基准） */
  async applyPlanCPU(basePositions: Float32Array, dabs: SculptDab[], precomputedFields?: Float32Array[]): Promise<SculptResult> {
    const t0 = performance.now();
    const n = this.mesh.vertexCount;
    const positions = new Float32Array(basePositions);
    const fields: Float32Array[] = precomputedFields ? [...precomputedFields] : [];
    let fieldCursor = 0;

    for (const dab of dabs) {
      let distance: Float32Array;
      if (precomputedFields && precomputedFields[fieldCursor]) {
        distance = precomputedFields[fieldCursor++];
      } else {
        const r = await this.geodesic.solve(dab.centerVertex);
        distance = r.distance;
        fields.push(distance);
      }
      const sign = dab.kind === 'pressure' ? -1 : 1;
      for (let i = 0; i < n; i++) {
        const w = gaussian(distance[i], dab.sigmaMm);
        const disp = sign * dab.depthMm * w;
        positions[i * 3] += this.normals[i * 3] * disp;
        positions[i * 3 + 1] += this.normals[i * 3 + 1] * disp;
        positions[i * 3 + 2] += this.normals[i * 3 + 2] * disp;
      }
      if (dab.kind === 'pressure' && dab.contralateral?.enabled) {
        let mirrorField: Float32Array;
        if (precomputedFields && precomputedFields[fieldCursor]) {
          mirrorField = precomputedFields[fieldCursor++];
        } else {
          const mirrorVertex = this.findContralateralVertex(dab.centerVertex);
          const r = await this.geodesic.solve(mirrorVertex);
          mirrorField = r.distance;
          fields.push(mirrorField);
        }
        const cfg = dab.contralateral;
        const sigma = dab.sigmaMm * (cfg.sigmaScale || 1.2);
        for (let i = 0; i < n; i++) {
          const w = gaussian(mirrorField[i], sigma);
          const disp = cfg.depthMm * w;
          positions[i * 3] += this.normals[i * 3] * disp;
          positions[i * 3 + 1] += this.normals[i * 3 + 1] * disp;
          positions[i * 3 + 2] += this.normals[i * 3 + 2] * disp;
        }
      }
    }
    return { positions, fields, gpuMs: performance.now() - t0 };
  }

  /**
   * 对侧顶点：以躯干 X 中轴面镜像，取最近顶点。
   * 解剖坐标系摆正后 X 即冠状横轴，镜像只翻 x 分量。
   */
  findContralateralVertex(vertex: number): number {
    const p = this.mesh.vertex(vertex);
    const target: Vec3 = [-p[0], p[1], p[2]];
    let best = vertex;
    let bestD = Infinity;
    const pos = this.mesh.positions;
    for (let i = 0; i < this.mesh.vertexCount; i++) {
      const dy = pos[i * 3 + 1] - target[1];
      if (Math.abs(dy) > 25) continue; // 同高度带
      const dx = pos[i * 3] - target[0];
      const dz = pos[i * 3 + 2] - target[2];
      const d = dx * dx + dy * dy + dz * dz;
      if (d < bestD) {
        bestD = d;
        best = i;
      }
    }
    return best;
  }

  /**
   * 修型前后逐顶点位移向量与沿法向的标量差值（mm），供红蓝色谱
   */
  diffField(before: Float32Array, after: Float32Array): { vectors: Float32Array; scalar: Float32Array } {
    const n = this.mesh.vertexCount;
    const vectors = new Float32Array(n * 3);
    const scalar = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const d = sub(
        [after[i * 3], after[i * 3 + 1], after[i * 3 + 2]],
        [before[i * 3], before[i * 3 + 1], before[i * 3 + 2]],
      );
      vectors.set(d, i * 3);
      const nrm: Vec3 = [this.normals[i * 3], this.normals[i * 3 + 1], this.normals[i * 3 + 2]];
      // 沿法向投影：内推为正（压垫红），外扩为负（释放蓝）
      scalar[i] = -dot3(d, nrm);
    }
    return { vectors, scalar };
  }

  get vertexNormals(): Float32Array {
    return this.normals;
  }
}

function dot3(a: Vec3, b: Vec3): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
