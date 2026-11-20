import { Mesh, faceCount, getVertex } from './mesh';
import { Vec3, v3 } from './math';

/**
 * 三角形 BVH（中位数分裂，叶子 4 个面）。
 * 用途：穿戴评估的体表-壳体最近点查询、壳厚抽点校验、自交检测。
 * 20 万面规模构建为 O(n log n)，查询 O(log n)。
 */

interface Node {
  min: Vec3;
  max: Vec3;
  left: number; // >=0 内部节点；-1 表示叶子
  right: number;
  faceStart: number;
  faceCount: number;
}

export interface NearestHit {
  face: number;
  point: Vec3;
  distance: number;
  normal: Vec3;
  /** 命中点是否从法线反侧接近（穿透判定用） */
  backside: boolean;
}

export class TriangleBVH {
  private nodes: Node[] = [];
  private faceOrder: Uint32Array;
  constructor(private mesh: Mesh) {
    const F = faceCount(mesh);
    this.faceOrder = new Uint32Array(F);
    for (let i = 0; i < F; i++) this.faceOrder[i] = i;
    this.build(0, F, 0);
  }

  private faceBounds(f: number): { min: Vec3; max: Vec3; center: Vec3 } {
    const min: Vec3 = [Infinity, Infinity, Infinity];
    const max: Vec3 = [-Infinity, -Infinity, -Infinity];
    for (let k = 0; k < 3; k++) {
      const v = this.mesh.indices[f * 3 + k];
      const p = getVertex(this.mesh, v);
      for (let c = 0; c < 3; c++) {
        min[c] = Math.min(min[c], p[c]);
        max[c] = Math.max(max[c], p[c]);
      }
    }
    return { min, max, center: v3.scale(v3.add(min, max), 0.5) };
  }

  private boundsOf(start: number, count: number): { min: Vec3; max: Vec3; center: Vec3 } {
    const min: Vec3 = [Infinity, Infinity, Infinity];
    const max: Vec3 = [-Infinity, -Infinity, -Infinity];
    for (let i = start; i < start + count; i++) {
      const b = this.faceBounds(this.faceOrder[i]);
      for (let c = 0; c < 3; c++) {
        min[c] = Math.min(min[c], b.min[c]);
        max[c] = Math.max(max[c], b.max[c]);
      }
    }
    return { min, max, center: v3.scale(v3.add(min, max), 0.5) };
  }

  private build(start: number, count: number, depth: number): number {
    const b = this.boundsOf(start, count);
    const nodeIdx = this.nodes.length;
    this.nodes.push({ min: b.min, max: b.max, left: -1, right: -1, faceStart: start, faceCount: count });
    if (count <= 4 || depth > 48) return nodeIdx;
    const extent = v3.sub(b.max, b.min);
    const axis = extent[0] >= extent[1] && extent[0] >= extent[2] ? 0 : extent[1] >= extent[2] ? 1 : 2;
    const slice = this.faceOrder.subarray(start, start + count);
    const arr = Array.from(slice);
    arr.sort((f1, f2) => this.faceBounds(f1).center[axis] - this.faceBounds(f2).center[axis]);
    slice.set(arr);
    const mid = start + (count >> 1);
    const left = this.build(start, mid - start, depth + 1);
    const right = this.build(mid, start + count - mid, depth + 1);
    this.nodes[nodeIdx].left = left;
    this.nodes[nodeIdx].right = right;
    this.nodes[nodeIdx].faceCount = 0;
    return nodeIdx;
  }

  private boxHitDist(min: Vec3, max: Vec3, p: Vec3, currentBest: number): number {
    let d2 = 0;
    for (let c = 0; c < 3; c++) {
      if (p[c] < min[c]) d2 += (min[c] - p[c]) ** 2;
      else if (p[c] > max[c]) d2 += (p[c] - max[c]) ** 2;
    }
    return d2 < currentBest * currentBest ? Math.sqrt(d2) : Infinity;
  }

  private triangleData(f: number): [Vec3, Vec3, Vec3] {
    return [
      getVertex(this.mesh, this.mesh.indices[f * 3]),
      getVertex(this.mesh, this.mesh.indices[f * 3 + 1]),
      getVertex(this.mesh, this.mesh.indices[f * 3 + 2]),
    ];
  }

  /** 点到三角形最近点（Ericson Real-Time Collision Detection 5.1.5） */
  private closestOnTriangle(p: Vec3, a: Vec3, b: Vec3, c: Vec3): { point: Vec3; normal: Vec3; backside: boolean } {
    const ab = v3.sub(b, a);
    const ac = v3.sub(c, a);
    const ap = v3.sub(p, a);
    const d1 = v3.dot(ab, ap);
    const d2 = v3.dot(ac, ap);
    if (d1 <= 0 && d2 <= 0) return { point: a, normal: this.faceNormal(a, b, c), backside: false };
    const bp = v3.sub(p, b);
    const d3 = v3.dot(ab, bp);
    const d4 = v3.dot(ac, bp);
    if (d3 >= 0 && d4 <= d3) return { point: b, normal: this.faceNormal(a, b, c), backside: false };
    const vc = d1 * d4 - d3 * d2;
    if (vc <= 0 && d1 >= 0 && d3 <= 0) {
      const v = d1 / (d1 - d3);
      return { point: v3.lerp(a, b, v), normal: this.faceNormal(a, b, c), backside: false };
    }
    const cp = v3.sub(p, c);
    const d5 = v3.dot(ab, cp);
    const d6 = v3.dot(ac, cp);
    if (d6 >= 0 && d5 <= d6) return { point: c, normal: this.faceNormal(a, b, c), backside: false };
    const vb = d5 * d2 - d1 * d6;
    if (vb <= 0 && d2 >= 0 && d6 <= 0) {
      const w = d2 / (d2 - d6);
      return { point: v3.lerp(a, c, w), normal: this.faceNormal(a, b, c), backside: false };
    }
    const va = d3 * d6 - d5 * d4;
    if (va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0) {
      const w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
      return { point: v3.lerp(b, c, w), normal: this.faceNormal(a, b, c), backside: false };
    }
    const denom = 1 / (va + vb + vc);
    const v = vb * denom;
    const w = vc * denom;
    const point = v3.add(a, v3.add(v3.scale(ab, v), v3.scale(ac, w)));
    const normal = this.faceNormal(a, b, c);
    const backside = v3.dot(v3.sub(p, point), normal) < 0;
    return { point, normal, backside };
  }

  private faceNormal(a: Vec3, b: Vec3, c: Vec3): Vec3 {
    return v3.norm(v3.cross(v3.sub(b, a), v3.sub(c, a)));
  }

  nearestPoint(p: Vec3): NearestHit {
    let best: NearestHit = { face: -1, point: [0, 0, 0], distance: Infinity, normal: [0, 0, 0], backside: false };
    const stack = [0];
    while (stack.length) {
      const node = this.nodes[stack.pop()!];
      if (this.boxHitDist(node.min, node.max, p, best.distance) === Infinity) continue;
      if (node.faceCount > 0) {
        for (let i = node.faceStart; i < node.faceStart + node.faceCount; i++) {
          const f = this.faceOrder[i];
          const [a, b, c] = this.triangleData(f);
          const hit = this.closestOnTriangle(p, a, b, c);
          const d = v3.dist(p, hit.point);
          if (d < best.distance) {
            best = { face: f, point: hit.point, distance: d, normal: hit.normal, backside: hit.backside };
          }
        }
      } else {
        stack.push(node.left, node.right);
      }
    }
    return best;
  }

  get nodeCount(): number {
    return this.nodes.length;
  }
}
