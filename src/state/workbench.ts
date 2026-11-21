// 扫描工作台：当前工作网格（原始/清洗/摆正）与标记点的内存态，页面间共享
import { TriangleMesh } from '../core/mesh';
import { AnatomicalLandmarks } from '../core/processing';

interface Workbench {
  rawMesh: TriangleMesh | null;
  cleanedMesh: TriangleMesh | null;
  alignedMesh: TriangleMesh | null;
  landmarks: AnatomicalLandmarks | null;
  cleaningStats: { removedComponents: number; removedFaces: number; filledHoles: number; filledTriangles: number } | null;
  filename: string;
  synthetic: boolean;
}

const wb: Workbench = {
  rawMesh: null,
  cleanedMesh: null,
  alignedMesh: null,
  landmarks: null,
  cleaningStats: null,
  filename: '',
  synthetic: false,
};

export const workbench = {
  reset() {
    wb.rawMesh = null;
    wb.cleanedMesh = null;
    wb.alignedMesh = null;
    wb.landmarks = null;
    wb.cleaningStats = null;
    wb.filename = '';
    wb.synthetic = false;
  },
  set raw(m: TriangleMesh | null) { wb.rawMesh = m; },
  get raw() { return wb.rawMesh; },
  set cleaned(m: TriangleMesh | null) { wb.cleanedMesh = m; },
  get cleaned() { return wb.cleanedMesh; },
  set aligned(m: TriangleMesh | null) { wb.alignedMesh = m; },
  get aligned() { return wb.alignedMesh; },
  set landmarksValue(lm: AnatomicalLandmarks | null) { wb.landmarks = lm; },
  get landmarksValue() { return wb.landmarks; },
  set stats(s: Workbench['cleaningStats']) { wb.cleaningStats = s; },
  get stats() { return wb.cleaningStats; },
  set name(n: string) { wb.filename = n; },
  get name() { return wb.filename; },
  set syntheticFlag(v: boolean) { wb.synthetic = v; },
  get syntheticFlag() { return wb.synthetic; },
  /** 当前用于修型的最终网格（优先摆正后的） */
  activeMesh(): TriangleMesh | null {
    return wb.alignedMesh ?? wb.cleanedMesh ?? wb.rawMesh;
  },
};
