// GPU 会话：设备/渲染器/网格与算法引擎的单例管理，页面间共享
import { initWebGPU, GpuContext } from '../gpu/context';
import { Renderer } from '../gpu/renderer';
import { TriangleMesh } from '../core/mesh';
import { HeatGeodesic } from '../gpu/geodesic';
import { SculptEngine } from '../core/sculpt';

interface Session {
  gpu: GpuContext;
  renderer: Renderer | null;
  baseMesh: TriangleMesh | null;       // 处理后的扫描网格
  sculptedMesh: TriangleMesh | null;   // 修型后网格
  shellMesh: TriangleMesh | null;
  geodesic: HeatGeodesic | null;
  sculpt: SculptEngine | null;
}

let session: Session | null = null;
const readyListeners = new Set<() => void>();

export async function ensureGPU(): Promise<GpuContext> {
  if (!session) {
    const gpu = await initWebGPU();
    session = { gpu, renderer: null, baseMesh: null, sculptedMesh: null, shellMesh: null, geodesic: null, sculpt: null };
    readyListeners.forEach((fn) => fn());
  }
  return session.gpu;
}

export function attachRenderer(canvas: HTMLCanvasElement): Renderer {
  if (!session) throw new Error('GPU 未初始化');
  if (!session.renderer) session.renderer = new Renderer(session.gpu.device, canvas);
  return session.renderer;
}

export function getRenderer(): Renderer | null {
  return session?.renderer ?? null;
}

export function setBaseMesh(mesh: TriangleMesh): HeatGeodesic {
  if (!session) throw new Error('GPU 未初始化');
  session.baseMesh = mesh;
  session.sculptedMesh = null;
  session.shellMesh = null;
  session.geodesic = new HeatGeodesic(session.gpu.device, mesh);
  session.sculpt = new SculptEngine(mesh, session.geodesic, session.gpu.device);
  return session.geodesic;
}

export function setSculptedMesh(mesh: TriangleMesh | null) {
  if (session) session.sculptedMesh = mesh;
}
export function setShellMesh(mesh: TriangleMesh | null) {
  if (session) session.shellMesh = mesh;
}

export function gpuSession(): Session {
  if (!session) throw new Error('GPU 未初始化，请先 ensureGPU()');
  return session;
}

export function onGPUReady(fn: () => void) {
  readyListeners.add(fn);
  return () => readyListeners.delete(fn);
}
