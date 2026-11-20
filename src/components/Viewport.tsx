import { useEffect, useRef } from 'react';
import { MeshRenderer, RenderMesh } from '../render/MeshRenderer';
import { vertexNormals, meshBounds, vertexCount } from '../geometry/mesh';
import type { Mesh } from '../geometry/mesh';

export interface ViewportMeshSpec {
  mesh: Mesh;
  id: string;
  alpha: number;
  baseColor: [number, number, number];
  useVertexColor?: boolean;
  visible: boolean;
}

interface ViewportProps {
  specs: ViewportMeshSpec[];
  /** 场景相机对准中心（默认自动取第一个网格包围盒中心） */
  focusKey?: string;
  height?: number;
}

/** 3D 视口：托管 WebGL2 MeshRenderer 生命周期，规格变化时重建 GPU 缓冲 */
export function Viewport({ specs, height = 560 }: ViewportProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rendererRef = useRef<MeshRenderer | null>(null);

  useEffect(() => {
    if (!canvasRef.current) return;
    const canvas = canvasRef.current;
    const resize = () => {
      const dpr = Math.min(2, window.devicePixelRatio || 1);
      canvas.width = canvas.clientWidth * dpr;
      canvas.height = height * dpr;
    };
    resize();
    let renderer: MeshRenderer;
    try {
      renderer = new MeshRenderer(canvas);
      rendererRef.current = renderer;
    } catch (e) {
      console.error(e);
      return;
    }
    rendererRef.current = renderer;
    const onResize = () => { resize(); renderer.render(); };
    window.addEventListener('resize', onResize);
    let raf = 0;
    const loop = () => {
      try { renderer.render(); } catch (e) { console.error('[Viewport] 渲染失败', e); }
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener('resize', onResize);
    };
  }, [height]);

  useEffect(() => {
    const renderer = rendererRef.current;
    if (!renderer) return;
    const renderMeshes: RenderMesh[] = specs
      .filter((s) => s.mesh && s.mesh.positions.length > 0)
      .map((s) => ({
        id: s.id,
        positions: s.mesh.positions,
        indices: s.mesh.indices,
        normals: ensureNormals(s.mesh),
        colors: s.mesh.colors,
        material: {
          alpha: s.alpha,
          baseColor: s.baseColor,
          useVertexColor: !!s.useVertexColor,
          visible: s.visible,
        },
      }));
    renderer.setMeshes(renderMeshes);
    if (renderMeshes.length > 0) {
      const first = specs.find((s) => s.mesh && s.mesh.positions.length)!.mesh;
      renderer.setCenter(meshBounds(first).center);
    }
  }, [specs]);

  return (
    <canvas
      ref={canvasRef}
      style={{ width: '100%', height, display: 'block', borderRadius: 8, background: '#1e2126' }}
    />
  );
}

const normalCache = new WeakMap<Mesh, Float32Array>();
function ensureNormals(mesh: Mesh): Float32Array {
  let n = normalCache.get(mesh);
  if (!n) {
    n = vertexNormals(mesh);
    normalCache.set(mesh, n);
  }
  return n;
}

/** 网格规模摘要（各页面复用） */
export function meshSummary(mesh?: Mesh): string {
  if (!mesh || vertexCount(mesh) === 0) return '无网格';
  return `${vertexCount(mesh).toLocaleString()} 顶点 / ${(mesh.indices.length / 3).toLocaleString()} 面`;
}
