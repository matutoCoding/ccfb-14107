// 通用 UI 助手：DOM 创建、Toast、Modal、SVG
export type ElProps = {
  class?: string;
  style?: string;
  dataset?: Record<string, string>;
} & Record<string, unknown>;

export function h<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  props: ElProps = {},
  children: Array<Node | string | null | false> = [],
): HTMLElementTagNameMap[K] {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (v === null || v === undefined) continue;
    if (k === 'class') el.className = v as string;
    else if (k === 'dataset') Object.assign(el.dataset, v);
    else if (k === 'style') el.setAttribute('style', v as string);
    else if (k.startsWith('on') && typeof v === 'function') {
      el.addEventListener(k.slice(2).toLowerCase(), v as EventListener);
    } else {
      (el as unknown as Record<string, unknown>)[k] = v;
    }
  }
  for (const c of children) {
    if (c === null || c === false || c === undefined) continue;
    el.append(typeof c === 'string' ? document.createTextNode(c) : c);
  }
  return el;
}

export function clear(el: HTMLElement) {
  el.replaceChildren();
}

const toastWrap = h('div', { class: 'toast-wrap' });
document.addEventListener('DOMContentLoaded', () => document.body.append(toastWrap));

export function toast(message: string, kind: 'info' | 'error' | 'success' = 'info', duration = 3200) {
  if (!toastWrap.parentElement) document.body.append(toastWrap);
  const el = h('div', { class: `toast ${kind === 'info' ? '' : kind}` }, [message]);
  toastWrap.append(el);
  setTimeout(() => el.remove(), duration);
}

export function modal(title: string, body: HTMLElement, onOk?: () => boolean | void): { close: () => void; el: HTMLElement } {
  const mask = h('div', { class: 'modal-mask' });
  const box = h('div', { class: 'modal' }, [
    h('h3', {}, [title]),
    body,
    h('div', { class: 'modal-actions' }, [
      h('button', { class: 'ghost', onClick: () => close() }, ['取消']),
      h('button', {
        class: 'primary',
        onClick: () => {
          if (onOk === undefined || onOk() !== false) close();
        },
      }, ['确定']),
    ]),
  ]);
  mask.append(box);
  mask.addEventListener('click', (e) => { if (e.target === mask) close(); });
  document.body.append(mask);
  function close() { mask.remove(); }
  return { close, el: box };
}

export function confirmDialog(message: string): Promise<boolean> {
  return new Promise((resolve) => {
    const mask = h('div', { class: 'modal-mask' });
    mask.append(h('div', { class: 'modal', style: 'width:400px' }, [
      h('h3', {}, ['请确认']),
      h('p', { class: 'muted' }, [message]),
      h('div', { class: 'modal-actions' }, [
        h('button', { class: 'ghost', onClick: () => { mask.remove(); resolve(false); } }, ['取消']),
        h('button', { class: 'danger', onClick: () => { mask.remove(); resolve(true); } }, ['确认删除']),
      ]),
    ]));
    document.body.append(mask);
  });
}

/** 折线图（SVG，无依赖） */
export function lineChart(series: Array<{ label: string; color: string; points: Array<[number, number]> }>, width = 560, height = 220) {
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
  svg.setAttribute('class', 'chart');
  const allX = series.flatMap((s) => s.points.map((p) => p[0]));
  const allY = series.flatMap((s) => s.points.map((p) => p[1]));
  const x0 = Math.min(...allX), x1 = Math.max(...allX, x0 + 1);
  const y0 = Math.min(...allY), y1 = Math.max(...allY, y0 + 1);
  const pad = 38;
  const sx = (x: number) => pad + ((x - x0) / (x1 - x0)) * (width - pad * 1.5);
  const sy = (y: number) => height - pad - ((y - y0) / (y1 - y0)) * (height - pad * 2);
  // 网格/轴
  for (let i = 0; i <= 4; i++) {
    const y = pad + ((height - pad * 2) / 4) * i;
    const val = y1 - ((y1 - y0) / 4) * i;
    svg.append(svgEl('line', { x1: pad, y1: y, x2: width - pad / 2, y2: y, stroke: '#2b3947', 'stroke-width': 1 }));
    svg.append(svgEl('text', { x: 6, y: y + 4, fill: '#8b98a5', 'font-size': 10 }, [`${val.toFixed(1)}°`]));
  }
  for (const s of series) {
    const d = s.points.map((p, i) => `${i ? 'L' : 'M'}${sx(p[0]).toFixed(1)},${sy(p[1]).toFixed(1)}`).join(' ');
    svg.append(svgEl('path', { d, fill: 'none', stroke: s.color, 'stroke-width': 2.2 }));
    for (const p of s.points) {
      svg.append(svgEl('circle', { cx: sx(p[0]), cy: sy(p[1]), r: 3.5, fill: s.color }));
      svg.append(svgEl('text', { x: sx(p[0]) - 8, y: sy(p[1]) - 8, fill: '#e6edf3', 'font-size': 10 }, [`${p[1].toFixed(1)}`]));
    }
  }
  // 图例
  series.forEach((s, i) => {
    svg.append(svgEl('rect', { x: pad + i * 130, y: 6, width: 12, height: 4, fill: s.color }));
    svg.append(svgEl('text', { x: pad + i * 130 + 16, y: 11, fill: '#8b98a5', 'font-size': 11 }, [s.label]));
  });
  return svg;
}

export function svgEl(tag: string, attrs: Record<string, string | number>, text: string[] = []) {
  const el = document.createElementNS('http://www.w3.org/2000/svg', tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, String(v));
  el.append(...text.map((t) => document.createTextNode(t)));
  return el;
}

export function download(filename: string, text: string, mime = 'text/plain') {
  const blob = new Blob([text], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = h('a', { href: url, download: filename });
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export function fmtBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 ** 2) return `${(n / 1024).toFixed(1)} KB`;
  if (n < 1024 ** 3) return `${(n / 1024 ** 2).toFixed(1)} MB`;
  return `${(n / 1024 ** 3).toFixed(2)} GB`;
}
