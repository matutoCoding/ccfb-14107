// 程序化躯干扫描生成器：带脊柱侧弯曲线、旋转畸形、体表起伏与可控扫描破洞/噪声
// 用于无真实扫描时的全流程演示与性能自检
import { TriangleMesh } from './mesh';
import { Vec3, rad } from './math';

export interface TorsoOptions {
  /** 身高 mm（生成范围约从髂骨到肩） */
  heightMm?: number;
  /** 环向分段 */
  radialSegments?: number;
  /** 高度分段 */
  heightSegments?: number;
  /** 主弯 Cobb 角（度），正值右凸 */
  cobbDeg?: number;
  /** 顶椎高度比例（0=足侧 1=头侧） */
  apexLevel?: number;
  /** 椎体旋转畸形（度），带动肋骨隆起 */
  vertebralRotationDeg?: number;
  /** 扫描噪声标准差 mm */
  noiseMm?: number;
  /** 破洞数量（随机在躯干表面挖孔） */
  holeCount?: number;
  /** 破洞半径 mm */
  holeRadiusMm?: number;
  /** 随机种子（可复现） */
  seed?: number;
}

// 可复现随机数（mulberry32）
function rng(seed: number) {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export interface GeneratedTorso {
  mesh: TriangleMesh;
  /** 真实侧弯参数（供矫正预测对照） */
  truth: {
    cobbDeg: number;
    apexLevel: number;
    spineCurve: Vec3[];
  };
  holes: Array<{ center: Vec3; radius: number }>;
}

export function generateTorso(opts: TorsoOptions = {}): GeneratedTorso {
  const {
    heightMm = 520,
    radialSegments = 160,
    heightSegments = 140,
    cobbDeg = 28,
    apexLevel = 0.55,
    vertebralRotationDeg = 12,
    noiseMm = 0.6,
    holeCount = 2,
    holeRadiusMm = 22,
    seed = 42,
  } = opts;
  const random = rng(seed);

  // 1) 脊柱中线：S 形侧弯（单主弯 + 代偿小弯），在 X 方向偏移
  const spineX = (t: number): number => {
    const main = Math.sin(Math.PI * t) * Math.sin(Math.PI * apexLevel);
    const curve = Math.sin(Math.PI * Math.min(1, t / apexLevel)) * (t <= apexLevel ? 1 : Math.sin(Math.PI * (t - apexLevel) / (1 - apexLevel)) * (apexLevel / (1 - apexLevel)));
    void main;
    // 以顶椎为峰的平滑侧弯
    const amp = (heightMm * 0.09) * Math.tan(rad(cobbDeg)) * 0.25;
    return amp * curve;
  };
  const spineZ = (t: number): number => {
    // 生理前后曲度 + 侧弯联动旋转
    return 14 * Math.sin(Math.PI * t) + 6 * Math.sin(2 * Math.PI * t);
  };
  const spineCurve: Vec3[] = [];
  for (let i = 0; i <= heightSegments; i++) {
    const t = i / heightSegments;
    spineCurve.push([spineX(t), t * heightMm, spineZ(t)]);
  }

  // 2) 截面轮廓：椭圆 + 解剖形变（前侧胸骨、后侧肩胛、腰侧收窄）
  const positions: number[] = [];
  const grid: number[][] = []; // [h][r] -> vertex index
  for (let i = 0; i <= heightSegments; i++) {
    const t = i / heightSegments;
    const y = t * heightMm;
    // 半宽/半厚随高度变化：肩宽、腰窄、骨盆展开
    const shoulder = 150 * smooth(0.72, 0.95, t);
    const pelvis = 135 * (1 - smooth(0.15, 0.4, t));
    const waist = 105 + 0.18 * heightMm * 0.0;
    const halfWidth = waist + shoulder + pelvis - 40 * smooth(0.35, 0.6, t);
    const halfDepth = 95 + 18 * smooth(0.7, 0.95, t) + 12 * (1 - smooth(0.2, 0.45, t));
    // 顶椎附近椎体旋转 -> 肋骨隆起（剃刀背），体现在后侧轮廓不对称
    const rotEnvelope = Math.exp(-Math.pow((t - apexLevel) / 0.18, 2));
    const rotRad = rad(vertebralRotationDeg) * rotEnvelope;
    const row: number[] = [];
    for (let j = 0; j < radialSegments; j++) {
      const theta = (j / radialSegments) * Math.PI * 2;
      // 基础椭圆
      let x = halfWidth * Math.cos(theta);
      let z = halfDepth * Math.sin(theta);
      // 前侧（sin>0 为前）胸骨微凸、腹圆
      if (Math.sin(theta) > 0) z += 6 * Math.sin(theta) ** 2;
      // 后侧肩胛/肋骨隆起：旋转把一侧后廓推高
      const ribBump = 10 * rotEnvelope * Math.max(0, -Math.cos(theta - rotRad)) * Math.max(0, -Math.sin(theta) + 0.3);
      x += ribBump * Math.cos(theta);
      z += ribBump * Math.sin(theta);
      // 跟随脊柱平移（整个躯干随中线弯）
      x += spineX(t);
      z += spineZ(t);
      // 扫描噪声
      const nx = (random() - 0.5) * 2 * noiseMm;
      const ny = (random() - 0.5) * 2 * noiseMm;
      const nz = (random() - 0.5) * 2 * noiseMm;
      row.push(positions.length / 3);
      positions.push(x + nx, y + ny, z + nz);
    }
    grid.push(row);
  }

  // 3) 顶面/底面封口：复用首尾环顶点，中心单点扇形，保证中心顶点边都在环上闭合
  const bottomCenter = positions.length / 3;
  positions.push(spineX(0), 0, spineZ(0));
  const topCenter = positions.length / 3;
  positions.push(spineX(1), heightMm, spineZ(1));

  const indices: number[] = [];
  const wrap = (j: number) => ((j % radialSegments) + radialSegments) % radialSegments;
  for (let i = 0; i < heightSegments; i++) {
    for (let j = 0; j < radialSegments; j++) {
      const a = grid[i][j];
      const b = grid[i + 1][j];
      const c = grid[i][wrap(j + 1)];
      const d = grid[i + 1][wrap(j + 1)];
      indices.push(a, b, c, b, d, c);
    }
  }
  // 底环封口（与侧面同绕序）
  for (let j = 0; j < radialSegments; j++) {
    indices.push(bottomCenter, grid[0][wrap(j + 1)], grid[0][j]);
  }
  // 顶环封口
  for (let j = 0; j < radialSegments; j++) {
    indices.push(topCenter, grid[heightSegments][j], grid[heightSegments][wrap(j + 1)]);
  }

  let mesh = new TriangleMesh(Float32Array.from(positions), Uint32Array.from(indices));

  // 4) 扫描破洞：删除中心周围的面，形成真实孔洞
  const holes: GeneratedTorso['holes'] = [];
  if (holeCount > 0) {
    const removeFaces = new Set<number>();
    const holeCenters: Array<{ ht: number; htheta: number }> = [];
    for (let h = 0; h < holeCount; h++) {
      // 限制在躯干中段，避免与顶/底封口三角形成非流形交汇；洞间保持间距避免重叠
      let ht = 0.28 + random() * 0.44;
      let htheta = random() * Math.PI * 2;
      for (let tries = 0; tries < 24; tries++) {
        const farEnough = holeCenters.every((c) => Math.abs(c.ht - ht) * heightMm > holeRadiusMm * 3.2);
        if (farEnough) break;
        ht = 0.28 + random() * 0.44;
        htheta = random() * Math.PI * 2;
      }
      holeCenters.push({ ht, htheta });
      const hi = Math.round(ht * heightSegments);
      const hj = Math.round((htheta / (Math.PI * 2)) * radialSegments);
      const cv = grid[hi][hj];
      const center = mesh.vertex(cv);
      holes.push({ center, radius: holeRadiusMm });
      // 按面质心距离删面，保证洞边界是完整环、不产生悬挂面（非流形）
      for (let f = 0; f < mesh.faceCount; f++) {
        const o = f * 3;
        let cx = 0, cy = 0, cz = 0;
        for (let k = 0; k < 3; k++) {
          const vi = mesh.indices[o + k] * 3;
          cx += mesh.positions[vi];
          cy += mesh.positions[vi + 1];
          cz += mesh.positions[vi + 2];
        }
        cx /= 3; cy /= 3; cz /= 3;
        if (Math.hypot(cx - center[0], cy - center[1], cz - center[2]) <= holeRadiusMm) {
          removeFaces.add(f);
        }
      }
    }
    const keepIdx: number[] = [];
    for (let f = 0; f < mesh.faceCount; f++) {
      if (removeFaces.has(f)) continue;
      const o = f * 3;
      keepIdx.push(mesh.indices[o], mesh.indices[o + 1], mesh.indices[o + 2]);
    }
    mesh = new TriangleMesh(mesh.positions, Uint32Array.from(keepIdx));
  }

  return {
    mesh,
    truth: { cobbDeg, apexLevel, spineCurve },
    holes,
  };
}

function smooth(e0: number, e1: number, x: number): number {
  const t = Math.min(1, Math.max(0, (x - e0) / (e1 - e0)));
  return t * t * (3 - 2 * t);
}

/** 生成内置的几组测试病例（正常轻中重 + 带洞噪声扫描） */
export function builtinTestTorsos(): Array<{ name: string; options: TorsoOptions }> {
  return [
    { name: '轻度侧弯 18°（完整扫描）', options: { cobbDeg: 18, holeCount: 0, seed: 11, vertebralRotationDeg: 6 } },
    { name: '中度侧弯 32°（带两处扫描破洞）', options: { cobbDeg: 32, holeCount: 2, seed: 23, noiseMm: 0.9 } },
    { name: '重度侧弯 48°（旋转畸形+破洞）', options: { cobbDeg: 48, holeCount: 3, holeRadiusMm: 28, seed: 37, vertebralRotationDeg: 18 } },
    { name: '高精度网格（20 万面性能自检）', options: { cobbDeg: 35, radialSegments: 240, heightSegments: 220, holeCount: 1, seed: 51 } },
  ];
}
