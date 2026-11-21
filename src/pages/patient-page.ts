// 患者档案页：档案 CRUD、复诊 Cobb/Risser/身高体重录入、角度变化折线、方案挂接
import { h, clear, modal, toast, confirmDialog, lineChart } from '../ui/dom';
import { store } from '../state/store';
import { CobbVisit, Patient } from '../core/types';

export function patientPage(): HTMLElement {
  const wrap = h('div', { class: 'grid', style: 'gap:14px' });
  const patient = store.state.currentPatient;

  if (!patient) {
    wrap.append(renderList());
    return wrap;
  }
  wrap.append(renderDetail(patient));
  return wrap;
}

function renderList(): HTMLElement {
  const card = h('div', { class: 'card' });
  const patients = store.state.patients;
  card.append(
    h('div', { class: 'row', style: 'justify-content:space-between;margin-bottom:14px' }, [
      h('h3', { style: 'margin:0' }, ['患者档案']),
      h('button', {
        class: 'primary',
        onClick: () => openPatientEditor(null),
      }, ['＋ 新建患者']),
    ]),
  );
  if (!patients.length) {
    card.append(h('div', { class: 'empty' }, ['暂无患者，点击“新建患者”开始；也可在扫描处理页生成程序化测试躯干。']));
    return card;
  }
  const table = h('table', {}, [
    h('thead', {}, [h('tr', {}, ['姓名', '性别', '诊断', '复诊次数', '最近 Cobb', '创建时间', '操作'].map((t) => h('th', {}, [t])))]),
    h('tbody', {}, patients.map((p) => {
      const last = p.visits[p.visits.length - 1];
      return h('tr', {}, [
        h('td', {}, [p.name]),
        h('td', {}, [p.gender]),
        h('td', {}, [p.diagnosis ?? '—']),
        h('td', {}, [String(p.visits.length)]),
        h('td', {}, [last ? `${last.cobbAngleDeg}°` : '—']),
        h('td', {}, [new Date(p.createdAt).toLocaleDateString('zh-CN')]),
        h('td', {}, [
          h('button', { class: 'ghost', onClick: () => { store.selectPatient(p); } }, ['打开']),
          ' ',
          h('button', { class: 'ghost', onClick: () => openPatientEditor(p) }, ['编辑']),
          ' ',
          h('button', {
            class: 'ghost danger',
            onClick: async () => {
              if (await confirmDialog(`确定删除患者「${p.name}」及其全部扫描、方案与壳体？`)) {
                await store.deletePatient(p.id);
                toast('已删除', 'success');
              }
            },
          }, ['删除']),
        ]),
      ]);
    })),
  ]);
  card.append(table);
  return card;
}

function renderDetail(patient: Patient): HTMLElement {
  const wrap = h('div', { class: 'grid cols-2' });

  // 左：基本信息 + 复诊表
  const info = h('div', { class: 'card' }, [
    h('div', { class: 'row', style: 'justify-content:space-between' }, [
      h('h3', {}, [`${patient.name} 的档案`]),
      h('div', { class: 'row' }, [
        h('button', { class: 'ghost', onClick: () => openPatientEditor(patient) }, ['编辑资料']),
        h('button', { class: 'ghost', onClick: () => store.selectPatient(null) }, ['返回列表']),
      ]),
    ]),
    h('div', { class: 'kv' }, [
      h('span', { class: 'k' }, ['性别']), h('span', {}, [patient.gender]),
      h('span', { class: 'k' }, ['出生日期']), h('span', {}, [patient.birthDate]),
      h('span', { class: 'k' }, ['联系方式']), h('span', {}, [patient.contact ?? '—']),
      h('span', { class: 'k' }, ['诊断']), h('span', {}, [patient.diagnosis ?? '—']),
    ]),
    h('div', { style: 'height:14px' }),
    h('div', { class: 'row', style: 'justify-content:space-between' }, [
      h('strong', {}, ['复诊记录（修型方案按复诊次数挂接）']),
      h('button', { class: 'primary', onClick: () => openVisitEditor(patient, null) }, ['＋ 新增复诊']),
    ]),
  ]);
  const visitTable = h('table', {}, [
    h('thead', {}, [h('tr', {}, ['次数', '日期', 'Cobb', '顶椎', 'Risser', '身高cm', '体重kg', '操作'].map((t) => h('th', {}, [t])))]),
    h('tbody', {}, patient.visits.length === 0
      ? [h('tr', {}, [h('td', { colSpan: 8, style: 'text-align:center;color:var(--text-dim)' }, ['暂无复诊记录'])])]
      : patient.visits.map((v, idx) => h('tr', {}, [
        h('td', {}, [`第 ${v.visitIndex} 次`]),
        h('td', {}, [v.date]),
        h('td', {}, [`${v.cobbAngleDeg}°`]),
        h('td', {}, [v.apexVertebra]),
        h('td', {}, [String(v.risserGrade)]),
        h('td', {}, [String(v.heightCm)]),
        h('td', {}, [String(v.weightKg)]),
        h('td', {}, [
          h('button', { class: 'ghost', onClick: () => openVisitEditor(patient, idx) }, ['编辑']),
          ' ',
          h('button', {
            class: 'ghost danger',
            onClick: async () => {
              if (await confirmDialog('删除该次复诊记录？')) {
                patient.visits.splice(idx, 1);
                await store.savePatient(patient);
                toast('已删除复诊', 'success');
              }
            },
          }, ['删除']),
        ]),
      ]))),
  ]);
  info.append(visitTable);

  // 右：Cobb 折线
  const chartCard = h('div', { class: 'card' }, [
    h('h3', {}, ['Cobb 角随复诊变化']),
  ]);
  if (patient.visits.length >= 1) {
    const sorted = [...patient.visits].sort((a, b) => a.visitIndex - b.visitIndex);
    chartCard.append(lineChart([
      {
        label: 'Cobb 角（度）',
        color: '#3d9df5',
        points: sorted.map((v) => [v.visitIndex, v.cobbAngleDeg]),
      },
    ]));
    chartCard.append(h('div', { class: 'muted', style: 'margin-top:10px' }, [
      `Risser 分级范围 0–5；当前 ${sorted[sorted.length - 1].risserGrade} 级。`,
      sorted.length >= 2
        ? ` 相邻复诊角度变化 ${(sorted[sorted.length - 1].cobbAngleDeg - sorted[sorted.length - 2].cobbAngleDeg).toFixed(1)}°。`
        : '',
    ]));
  } else {
    chartCard.append(h('div', { class: 'empty' }, ['录入复诊后自动绘制角度折线']));
  }

  wrap.append(info, chartCard);
  return wrap;
}

function openPatientEditor(existing: Patient | null) {
  const p: Patient = existing
    ? { ...existing }
    : {
        id: store.newId('pat'),
        name: '',
        gender: '女',
        birthDate: '2010-01-01',
        contact: '',
        diagnosis: '青少年特发性脊柱侧弯',
        createdAt: Date.now(),
        visits: [],
      };
  const body = h('div', {}, [
    field('姓名', input('text', p.name, (v) => (p.name = v))),
    field('性别', selectInput(['女', '男'], p.gender, (v) => (p.gender = v as Patient['gender']))),
    field('出生日期', input('date', p.birthDate, (v) => (p.birthDate = v))),
    field('联系方式', input('text', p.contact ?? '', (v) => (p.contact = v))),
    field('诊断', input('text', p.diagnosis ?? '', (v) => (p.diagnosis = v))),
  ]);
  modal(existing ? '编辑患者' : '新建患者', body, () => {
    if (!p.name.trim()) { toast('请填写姓名', 'error'); return false; }
    void store.savePatient(p);
    if (!existing) store.selectPatient(p);
    toast('已保存', 'success');
  });
}

function openVisitEditor(patient: Patient, index: number | null) {
  const nextIndex = patient.visits.length ? Math.max(...patient.visits.map((v) => v.visitIndex)) + 1 : 1;
  const v: CobbVisit = index !== null && patient.visits[index]
    ? { ...patient.visits[index] }
    : {
        id: store.newId('vis'),
        date: new Date().toISOString().slice(0, 10),
        visitIndex: nextIndex,
        cobbAngleDeg: 25,
        apexVertebra: 'T8',
        risserGrade: 2,
        heightCm: 155,
        weightKg: 45,
        note: '',
      };
  const body = h('div', {}, [
    field('复诊次数', input('number', String(v.visitIndex), (x) => (v.visitIndex = Number(x)))),
    field('日期', input('date', v.date, (x) => (v.date = x))),
    field('Cobb 角（度）', input('number', String(v.cobbAngleDeg), (x) => (v.cobbAngleDeg = Number(x)))),
    field('顶椎节段', input('text', v.apexVertebra, (x) => (v.apexVertebra = x))),
    field('Risser 分级（0–5）', input('number', String(v.risserGrade), (x) => (v.risserGrade = Math.min(5, Math.max(0, Number(x)))))),
    field('身高 cm', input('number', String(v.heightCm), (x) => (v.heightCm = Number(x)))),
    field('体重 kg', input('number', String(v.weightKg), (x) => (v.weightKg = Number(x)))),
    field('备注', input('text', v.note ?? '', (x) => (v.note = x))),
  ]);
  modal(index === null ? '新增复诊' : '编辑复诊', body, () => {
    if (index !== null && patient.visits[index]) patient.visits[index] = v;
    else patient.visits.push(v);
    patient.visits.sort((a, b) => a.visitIndex - b.visitIndex);
    void store.savePatient(patient);
    toast('复诊已保存', 'success');
  });
}

function field(label: string, control: HTMLElement) {
  return h('div', { class: 'field' }, [h('label', {}, [label]), control]);
}
function input(type: string, value: string, on: (v: string) => void) {
  const el = h('input', { type, value });
  el.addEventListener('input', () => on(el.value));
  return el;
}
function selectInput(options: string[], value: string, on: (v: string) => void) {
  const el = h('select', {}, options.map((o) => h('option', { value: o, selected: o === value }, [o])));
  el.addEventListener('change', () => on(el.value));
  return el;
}
void clear;
