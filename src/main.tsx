import React, { useReducer, useEffect } from 'react';
import { createRoot } from 'react-dom/client';
import { AppContext, initialState, reducer } from './state/appState';
import { AppShell } from './App';
import { gpuCompute } from './gpu/GpuCompute';
import { db } from './db/database';
import { DEFAULT_SETTINGS } from './db/schema';
import './styles.css';

function Root() {
  const [state, dispatch] = useReducer(reducer, initialState);

  useEffect(() => {
    // 启动：初始化 WebGPU、载入设置与患者列表
    (async () => {
      const cap = await gpuCompute.init();
      dispatch({ type: 'setGpu', gpu: cap });
      const settings = await db.getSettings();
      dispatch({ type: 'setSettings', settings: settings ?? DEFAULT_SETTINGS });
      const patients = await db.getAll<import('./db/schema').Patient>('patients');
      patients.sort((a, b) => b.updatedAt - a.updatedAt);
      dispatch({ type: 'setPatients', patients });
    })();
  }, []);

  // 轻提示自动消失
  useEffect(() => {
    if (!state.toast) return;
    const t = setTimeout(() => dispatch({ type: 'toast' }), 2600);
    return () => clearTimeout(t);
  }, [state.toast]);

  return (
    <AppContext.Provider value={{ state, dispatch }}>
      <AppShell />
      {state.toast && <div className="toast">{state.toast}</div>}
    </AppContext.Provider>
  );
}

createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <Root />
  </React.StrictMode>,
);
