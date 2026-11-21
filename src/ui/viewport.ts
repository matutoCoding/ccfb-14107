// 共享 3D 视口：挂载 WebGPU canvas、渲染循环、HUD 与工具条插槽
import { h } from './dom';
import { attachRenderer, getRenderer } from '../state/gpu-session';

export interface ViewportHandle {
  el: HTMLElement;
  canvas: HTMLCanvasElement;
  hud: HTMLElement;
  tools: HTMLElement;
  renderer: ReturnType<typeof attachRenderer>;
  stop: () => void;
}

export function createViewport(height = 560): ViewportHandle {
  const canvas = h('canvas', {});
  const hud = h('div', { class: 'viewport-hud' }, ['拖拽旋转 · 滚轮缩放']);
  const tools = h('div', { class: 'viewport-tools' });
  const el = h('div', { class: 'viewport', style: `min-height:${height}px` }, [canvas, tools, hud]);
  const renderer = attachRenderer(canvas);
  let raf = 0;
  let stopped = false;
  const loop = () => {
    if (stopped) return;
    if (!document.body.contains(canvas)) { // 页面切换后元素脱离 DOM，自动停帧
      stop();
      return;
    }
    renderer.render();
    raf = requestAnimationFrame(loop);
  };
  raf = requestAnimationFrame(loop);
  function stop() {
    stopped = true;
    cancelAnimationFrame(raf);
  }
  return {
    el,
    canvas,
    hud,
    tools,
    renderer,
    stop,
  };
}

export function requireRenderer() {
  const r = getRenderer();
  if (!r) throw new Error('渲染器尚未初始化');
  return r;
}
