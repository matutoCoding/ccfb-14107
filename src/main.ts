import './styles.css';
import { h, clear, toast } from './ui/dom';
import { store } from './state/store';
import { ensureGPU } from './state/gpu-session';
import { patientPage } from './pages/patient-page';
import { scanPage } from './pages/scan-page';
import { sculptPage } from './pages/sculpt-page';
import { shellPage } from './pages/shell-page';
import { fitPage } from './pages/fit-page';
import { settingsPage } from './pages/settings-page';

export type Route = 'patients' | 'scan' | 'sculpt' | 'shell' | 'fit' | 'settings';

const NAV: Array<{ route: Route; label: string; icon: string }> = [
  { route: 'patients', label: '患者档案', icon: '①' },
  { route: 'scan', label: '扫描处理', icon: '②' },
  { route: 'sculpt', label: '修型', icon: '③' },
  { route: 'shell', label: '支具壳体', icon: '④' },
  { route: 'fit', label: '穿戴评估', icon: '⑤' },
  { route: 'settings', label: '设置', icon: '⚙' },
];

const PAGE_TITLES: Record<Route, string> = {
  patients: '患者档案',
  scan: '扫描处理',
  sculpt: '修型设计',
  shell: '支具壳体',
  fit: '穿戴评估',
  settings: '系统设置',
};

class App {
  route: Route = 'patients';
  private contentEl!: HTMLElement;
  private topbarEl!: HTMLElement;

  async boot() {
    const root = document.getElementById('app')!;
    const shell = h('div', { class: 'app-shell' }, [
      h('nav', { class: 'sidebar' }, [
        h('div', { class: 'logo' }, ['色努支具工作台', h('small', {}, ['WebGPU 数字化设计系统'])]),
        ...NAV.map((n) =>
          h('div', {
            class: `nav-item ${n.route === this.route ? 'active' : ''}`,
            dataset: { route: n.route },
            onClick: () => this.navigate(n.route),
          }, [h('span', { class: 'step' }, [n.icon]), n.label]),
        ),
        h('div', { class: 'sidebar-footer' }, [
          '数据仅存于本机 IndexedDB\n定稿导出 STL 供 3D 打印',
        ]),
      ]),
      h('div', { class: 'main' }, [
        h('div', { class: 'topbar' }, [
          (this.topbarEl = h('h2', {}, [PAGE_TITLES[this.route]])),
          h('span', { class: 'crumb', id: 'crumb' }),
        ]),
        (this.contentEl = h('div', { class: 'content' })),
      ]),
    ]);
    root.append(shell);

    try {
      await ensureGPU();
      toast('WebGPU 设备初始化成功', 'success');
    } catch (e) {
      toast(`WebGPU 不可用：${(e as Error).message}。计算功能将受限。`, 'error', 8000);
    }
    await store.init();
    this.render();
  }

  navigate(route: Route) {
    this.route = route;
    document.querySelectorAll<HTMLElement>('.nav-item').forEach((el) => {
      el.classList.toggle('active', el.dataset.route === route);
    });
    this.render();
  }

  private render() {
    clear(this.contentEl);
    this.topbarEl.textContent = PAGE_TITLES[this.route];
    const crumb = document.getElementById('crumb');
    if (crumb) {
      const p = store.state.currentPatient;
      crumb.textContent = p ? `当前患者：${p.name}` : '';
    }
    const pages: Record<Route, () => HTMLElement | Promise<HTMLElement>> = {
      patients: patientPage,
      scan: scanPage,
      sculpt: sculptPage,
      shell: shellPage,
      fit: fitPage,
      settings: settingsPage,
    };
    Promise.resolve(pages[this.route]()).then((el) => this.contentEl.append(el)).catch((e) => {
      console.error(e);
      this.contentEl.append(h('div', { class: 'card' }, [`页面渲染失败：${(e as Error).message}`]));
    });
  }
}

new App().boot();
