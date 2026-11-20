import { useApp, PageKey } from './state/appState';
import { PatientsPage } from './pages/PatientsPage';
import { ScanPage } from './pages/ScanPage';
import { SculptPage } from './pages/SculptPage';
import { ShellPage } from './pages/ShellPage';
import { FitPage } from './pages/FitPage';
import { SettingsPage } from './pages/SettingsPage';

const NAV: { key: PageKey; label: string; icon: string }[] = [
  { key: 'patients', label: '患者档案', icon: '🧑‍⚕️' },
  { key: 'scan', label: '扫描处理', icon: '📷' },
  { key: 'sculpt', label: '修型', icon: '✏️' },
  { key: 'shell', label: '支具壳体', icon: '🧱' },
  { key: 'fit', label: '穿戴评估', icon: '📏' },
  { key: 'settings', label: '设置', icon: '⚙️' },
];

export function AppShell() {
  const { state, dispatch } = useApp();
  return (
    <div className="app-shell">
      <nav className="sidebar">
        <h1>色努支具<br />数字化设计工作台</h1>
        {NAV.map((n) => (
          <button
            key={n.key}
            className={`nav-item ${state.page === n.key ? 'active' : ''}`}
            onClick={() => dispatch({ type: 'setPage', page: n.key })}
          >
            <span>{n.icon}</span>
            <span>{n.label}</span>
          </button>
        ))}
        <div className="gpu-badge">
          {state.gpu.available ? (
            <><span className="on">● WebGPU 已就绪</span><br />通用计算走 GPU 着色器</>
          ) : (
            <><span className="off">● WebGPU 不可用</span><br />{state.gpu.reason ?? 'CPU 兜底运行中'}</>
          )}
        </div>
      </nav>
      <main className="main">
        {state.page === 'patients' && <PatientsPage />}
        {state.page === 'scan' && <ScanPage />}
        {state.page === 'sculpt' && <SculptPage />}
        {state.page === 'shell' && <ShellPage />}
        {state.page === 'fit' && <FitPage />}
        {state.page === 'settings' && <SettingsPage />}
      </main>
    </div>
  );
}
