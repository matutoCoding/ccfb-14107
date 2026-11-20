import { Mesh } from './mesh';

export interface SyntheticTorsoOptions {
  /** 环向分段 */
  segments?: number;
  /** 高度分段（决定面数规模） */
  rings?: number;
  /** 身高（毫米），决定躯干高度 */
  heightMm?: number;
  /** Cobb 角（度），驱动冠状面侧弯 */
  cobbDeg?: number;
  /** 顶椎高度比例 0..1 */
  apexRatio?: number;
  /** 凸侧：1 左凸（+X），-1 右凸 */
  convexSide?: 1 | -1;
  /** 扫描破洞数量 */
  holes?: number;
  /** 噪声幅度（毫米，模拟扫描抖动） */
  noiseMm?: number;
  /** 是否附带手臂残段（测清理流程） */
  withArmStubs?: boolean;
  /** 是否附带扶架杂点 */
  withFrameJunk?: boolean;
  seed?: number;
}

/** 可复现随机数（mulberry32） */
function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * 生成参数化躯干扫描：
 * 椭圆截面沿高度变化（胸/腰/骨盆），截面中心按 Cobb 角走 S/C 形曲线，
 * 凸侧叠加肋骨隆起，可选破洞、噪声、手臂残段和扶架杂点。
 */
export function generateTorso(opts: SyntheticTorsoOptions = {}): Mesh {
  const seg = opts.segments ?? 160;
  const rings = opts.rings ?? 120;
  const H = opts.heightMm ?? 720;
  const cobb = (opts.cobbDeg ?? 28) * Math.PI / 180;
  const apexR = opts.apexRatio ?? 0.55;
  const side = opts.convexSide ?? 1;
  const rand = rng(opts.seed ?? 42);

  const positions: number[] = [];
  const indices: number[] = [];
  // 截面半径曲线（半宽 rx、厚 rz）随高度变化
  const radii = (t: number): [number, number] => {
    // 骨盆宽、腰部收、胸廓展、肩部收
    const rx = 130 + 45 * gauss(t, 0.12, 0.1) - 30 * gauss(t, 0.45, 0.09) + 38 * gauss(t, 0.72, 0.12) - 18 * gauss(t, 0.95, 0.06);
    const rz = 95 + 18 * gauss(t, 0.12, 0.1) - 16 * gauss(t, 0.45, 0.09) + 14 * gauss(t, 0.7, 0.12);
    return [rx, rz];
  };
  // 脊柱侧弯：截面中心横向偏移，C 形（单弧），偏移幅度 ≈ H·sin(cobb)/4 的几何近似
  const curveAmp = H * Math.sin(cobb) * 0.22;
  const centerX = (t: number) => side * curveAmp * Math.sin(Math.PI * t) * (t < apexR ? t / apexR : (1 - t) / (1 - apexR));
  // 顶椎附近肋骨隆起（剃刀背）：凸侧背外局部鼓包
  const ribHump = (t: number, theta: number) => {
    const wt = Math.exp(-((t - apexR) ** 2) / (2 * 0.12 ** 2));
    const wth = Math.exp(-((theta - (side > 0 ? 0 : Math.PI)) ** 2) / (2 * 0.5 ** 2));
    return 18 * wt * wth;
  };

  const id = (r: number, s: number) => r * seg + (s % seg);

  for (let r = 0; r <= rings; r++) {
    const t = r / rings;
    const y = (t - 0.5) * H;
    const [rx, rz] = radii(t);
    const cx = centerX(t);
    for (let s = 0; s < seg; s++) {
      const theta = (s / seg) * Math.PI * 2;
      const hump = ribHump(t, theta);
      let x = cx + (rx + hump) * Math.cos(theta);
      let z = (rz + hump * 0.5) * Math.sin(theta);
      if (opts.noiseMm) {
        x += (rand() - 0.5) * 2 * opts.noiseMm;
        z += (rand() - 0.5) * 2 * opts.noiseMm;
      }
      positions.push(x, y, z);
    }
  }
  for (let r = 0; r < rings; r++) {
    for (let s = 0; s < seg; s++) {
      const a = id(r, s), b = id(r, s + 1), c = id(r + 1, s), d = id(r + 1, s + 1);
      indices.push(a, c, b, b, c, d);
    }
  }

  let mesh: Mesh = { positions: new Float32Array(positions), indices: new Uint32Array(indices), name: 'synthetic-torso' };

  // 破洞：随机删若干环形面片块（模拟扫描缺失）
  if (opts.holes) {
    const alive = new Uint8Array(mesh.indices.length / 3).fill(1);
    for (let h = 0; h < opts.holes; h++) {
      const cr = 4 + Math.floor(rand() * (rings * 0.15));
      const cs = Math.floor(rand() * seg);
      const rw = 3 + Math.floor(rand() * 4);
      const sw = 4 + Math.floor(rand() * 6);
      for (let r = cr; r < Math.min(rings, cr + rw); r++) {
        for (let s = cs; s < cs + sw; s++) {
          const faceBase = (r * seg + (s % seg)) * 2;
          alive[faceBase] = 0;
          alive[faceBase + 1] = 0;
        }
      }
    }
    mesh = filterFaces(mesh, alive);
  }

  // 手臂残段：两侧上方加小圆柱连到躯干（连通但径向突出）
  if (opts.withArmStubs) {
    mesh = appendArmStubs(mesh, H, rand);
  }
  // 扶架杂点：孤立小三角片/点云碎片
  if (opts.withFrameJunk) {
    mesh = appendFrameJunk(mesh, H, rand);
  }
  return mesh;
}

function gauss(x: number, mu: number, sigma: number): number {
  return Math.exp(-((x - mu) ** 2) / (2 * sigma * sigma));
}

function filterFaces(m: Mesh, alive: Uint8Array): Mesh {
  const idx: number[] = [];
  for (let f = 0; f < alive.length; f++) {
    if (alive[f]) idx.push(m.indices[f * 3], m.indices[f * 3 + 1], m.indices[f * 3 + 2]);
  }
  return { positions: m.positions, indices: new Uint32Array(idx), name: m.name };
}

function appendArmStubs(m: Mesh, H: number, rand: () => number): Mesh {
  const pos = Array.from(m.positions);
  const idx = Array.from(m.indices);
  for (const side of [-1, 1]) {
    const base = pos.length / 3;
    const n = 10;
    const x0 = side * 150, y0 = H * 0.28, z0 = 0;
    for (let i = 0; i <= n; i++) {
      const t = i / n;
      for (let s = 0; s < 8; s++) {
        const th = (s / 8) * Math.PI * 2;
        const r = 26 - t * 4;
        pos.push(
          x0 + side * t * 90 + r * Math.cos(th),
          y0 - t * 180,
          z0 + r * Math.sin(th),
        );
      }
    }
    for (let i = 0; i < n; i++) {
      for (let s = 0; s < 8; s++) {
        const a = base + i * 8 + s;
        const b = base + i * 8 + ((s + 1) % 8);
        const c = base + (i + 1) * 8 + s;
        const d = base + (i + 1) * 8 + ((s + 1) % 8);
        idx.push(a, c, b, b, c, d);
      }
    }
  }
  void rand;
  return { positions: new Float32Array(pos), indices: new Uint32Array(idx), name: m.name };
}

function appendFrameJunk(m: Mesh, H: number, rand: () => number): Mesh {
  const pos = Array.from(m.positions);
  const idx = Array.from(m.indices);
  // 3 个漂浮的小三角片 + 两根细杆（每根 6 个窄面）
  for (let k = 0; k < 3; k++) {
    const base = pos.length / 3;
    const cx = (rand() - 0.5) * 420;
    const cy = (rand() - 0.5) * H;
    const cz = -180 - rand() * 80;
    const s = 8 + rand() * 10;
    for (let i = 0; i < 3; i++) {
      const th = (i / 3) * Math.PI * 2;
      pos.push(cx + s * Math.cos(th), cy + s * Math.sin(th), cz);
    }
    idx.push(base, base + 1, base + 2);
  }
  for (let k = 0; k < 2; k++) {
    const base = pos.length / 3;
    const x = (rand() - 0.5) * 500;
    const z = -220;
    const y1 = -H * 0.4, y2 = H * 0.4;
    pos.push(x, y1, z, x + 6, y1, z, x, y2, z, x + 6, y2, z);
    idx.push(base, base + 2, base + 1, base + 1, base + 2, base + 3);
  }
  return { positions: new Float32Array(pos), indices: new Uint32Array(idx), name: m.name };
}

/** 预设测试组：覆盖不同面数、Cobb、破洞与噪声 */
export function presetScans(): { name: string; options: SyntheticTorsoOptions }[] {
  return [
    { name: '标准侧弯躯干（约6万面，Cobb 28°）', options: { segments: 100, rings: 75, cobbDeg: 28, seed: 1 } },
    { name: '性能用密网格（约20万面，Cobb 35°）', options: { segments: 160, rings: 160, cobbDeg: 35, seed: 2 } },
    { name: '带破洞扫描（3 处缺失 + 噪声）', options: { segments: 120, rings: 90, cobbDeg: 22, holes: 3, noiseMm: 1.2, seed: 3 } },
    { name: '含手臂与扶架杂点（清理流程测试）', options: { segments: 110, rings: 85, cobbDeg: 31, withArmStubs: true, withFrameJunk: true, seed: 4 } },
    { name: '重度侧弯右凸（Cobb 52°，破洞 2 处）', options: { segments: 130, rings: 100, cobbDeg: 52, convexSide: -1, holes: 2, seed: 5 } },
  ];
}
