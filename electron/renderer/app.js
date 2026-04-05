'use strict';
/* ─────────────────────────────────────────────────────────────────────────────
   app.js — AutoNote renderer
   Single-file SPA: state, terminal, navigation, and all 7 pages.
───────────────────────────────────────────────────────────────────────────── */

// ── Global state ──────────────────────────────────────────────────────────────
const State = {
  courses:    [],     // [{ id, name }]
  currentPage: 0,
  running:    false,
  pythonPath: null,
  outputDir:  null,
  dataDir:    null,
  // Persisted Pipeline page form values (survive navigation)
  pipeline: {
    courseId:   '',
    detail:     '7',
    courseName: '',
    lecFilter:  '',
    language:   'en',
    stealth:    false,
    force:      false,
    steps: { mat: true, vid: true, trans: true, align: true, gen: true },
  },
};

// ── Snackbar ──────────────────────────────────────────────────────────────────
let snackTimer = null;
function snack(msg, ok = true) {
  let el = document.getElementById('snackbar');
  if (!el) {
    el = document.createElement('div');
    el.id = 'snackbar';
    document.body.appendChild(el);
  }
  el.textContent = msg;
  el.className   = 'show ' + (ok ? 'ok' : 'error');
  clearTimeout(snackTimer);
  snackTimer = setTimeout(() => { el.className = ''; }, 2800);
}

// ── Terminal ──────────────────────────────────────────────────────────────────
const Term = (() => {
  const MAX_LINES    = 2000;
  const ANSI_RE      = /\x1b\[[0-9;]*[mABCDEFGHJKSTfr]|\x1b\][^\x07]*\x07|\x1b[()][AB012]/g;
  let lines          = [];   // array of {text, cls} — committed (newline-terminated)
  let currentText    = '';   // current partial line buffer
  let scheduledRender = false;
  let renderedCount  = 0;    // lines[0..renderedCount-1] are already in the DOM
  let _lastWasCR     = false; // cross-chunk CRLF state (Windows \r\n handling)

  function classify(text) {
    const t = text.toLowerCase().trimStart();
    if (t.startsWith('error') || t.startsWith('traceback') || t.startsWith('exception')) return 'err';
    if (t.startsWith('warning')) return 'warn';
    if (t.startsWith('$ '))      return 'cmd';
    if (t.startsWith('✓') || t.includes('completed successfully')) return 'ok';
    return '';
  }

  function scheduleRender() {
    if (!scheduledRender) {
      scheduledRender = true;
      requestAnimationFrame(() => {
        scheduledRender = false;
        render();
      });
    }
  }

  function render() {
    const el = document.getElementById('terminal-output');
    if (!el) return;
    const atBottom = el.scrollHeight - el.clientHeight - el.scrollTop < 30;

    // If the DOM was wiped (after clear()), reset tracked count
    if (el.children.length === 0) renderedCount = 0;

    // Prune orphaned nodes from the front (happen when MAX_LINES rolls lines off)
    // DOM should have exactly renderedCount nodes + 1 partial placeholder
    const expectedDOMNodes = renderedCount + 1;
    if (el.children.length > expectedDOMNodes) {
      const excess = el.children.length - expectedDOMNodes;
      for (let i = 0; i < excess; i++) el.removeChild(el.firstChild);
    }

    // Append only newly committed lines (don't touch existing DOM nodes)
    if (renderedCount < lines.length) {
      const frag = document.createDocumentFragment();
      for (let i = renderedCount; i < lines.length; i++) {
        const { text, cls } = lines[i];
        const div = document.createElement('div');
        div.className = 'term-line' + (cls ? ' ' + cls : '');
        div.textContent = text;
        frag.appendChild(div);
      }
      // Insert before the partial-line placeholder if it exists
      const partDiv = el.lastChild;
      if (partDiv && partDiv._isPartial) {
        el.insertBefore(frag, partDiv);
      } else {
        el.appendChild(frag);
      }
      renderedCount = lines.length;
    }

    // Update the partial-line placeholder in-place (no flicker)
    let partDiv = el.lastChild;
    if (!partDiv || !partDiv._isPartial) {
      partDiv = document.createElement('div');
      partDiv.className = 'term-line';
      partDiv._isPartial = true;
      el.appendChild(partDiv);
    }
    partDiv.textContent = currentText;

    if (atBottom) el.scrollTop = el.scrollHeight;
  }

  function commitLine(t) {
    if (t.startsWith('<frozen importlib')) return;
    lines.push({ text: t, cls: classify(t) });
    if (lines.length > MAX_LINES) {
      lines.shift();
      renderedCount = Math.max(0, renderedCount - 1);
    }
  }

  function process(raw) {
    const text = raw.replace(ANSI_RE, '');
    for (let i = 0; i < text.length; i++) {
      const ch = text[i];
      if (_lastWasCR) {
        _lastWasCR = false;
        if (ch === '\n') {
          // \r\n (Windows CRLF) — commit the line built before the \r
          commitLine(currentText);
          currentText = '';
          continue;
        } else {
          // Standalone \r — progress-bar overwrite; discard current partial
          currentText = '';
          // fall through to process ch
        }
      }
      if (ch === '\r') {
        _lastWasCR = true; // defer: need to see if \n follows
      } else if (ch === '\n') {
        commitLine(currentText);
        currentText = '';
      } else {
        currentText += ch;
      }
    }
    scheduleRender();
  }

  function write(text, cls = '') {
    for (const line of text.split('\n')) {
      lines.push({ text: line, cls });
      if (lines.length > MAX_LINES) {
        lines.shift();
        renderedCount = Math.max(0, renderedCount - 1);
      }
    }
    scheduleRender();
  }

  function clear() {
    lines = [];
    currentText = '';
    renderedCount = 0;
    _lastWasCR = false;
    const el = document.getElementById('terminal-output');
    if (el) el.innerHTML = '';
    setStatus('');
  }

  function setStatus(msg, cls = '') {
    const el = document.getElementById('term-status');
    if (el) { el.textContent = msg; el.style.color = cls || 'var(--c-primary)'; }
  }

  function showStop(visible) {
    const btn = document.getElementById('term-stop');
    if (btn) btn.style.display = visible ? 'inline-flex' : 'none';
  }

  return { process, write, clear, setStatus, showStop };
})();

// ── Icons (inline SVG helpers) ────────────────────────────────────────────────
const I = {
  dashboard:  '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/></svg>',
  pipeline:   '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="5" r="2"/><circle cx="12" cy="12" r="2"/><circle cx="12" cy="19" r="2"/><line x1="12" y1="7" x2="12" y2="10"/><line x1="12" y1="14" x2="12" y2="17"/></svg>',
  download:   '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>',
  transcribe: '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" y1="19" x2="12" y2="23"/><line x1="8" y1="23" x2="16" y2="23"/></svg>',
  align:      '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"/><path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"/></svg>',
  generate:   '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><polyline points="10 9 9 9 8 9"/></svg>',
  settings:   '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>',
  refresh:    '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="23 4 23 10 17 10"/><polyline points="1 20 1 14 7 14"/><path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"/></svg>',
  play:       '<svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"/></svg>',
  info:       '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>',
  folder:     '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"/></svg>',
  video:      '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="23 7 16 12 23 17 23 7"/><rect x="1" y="5" width="15" height="14" rx="2"/></svg>',
};

// ── Shared UI helpers ──────────────────────────────────────────────────────────
function esc(s) {
  return String(s ?? '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

function chip(label, cls) {
  return `<span class="chip ${cls}">${esc(label)}</span>`;
}

function statusChip(done, total) {
  if (total === 0) return chip('○ none',       'chip-none');
  if (done >= total) return chip(`✓ ${done}/${total}`, 'chip-success');
  return chip(`◐ ${done}/${total}`, 'chip-warn');
}

function courseOptions(includeAll = false, selectedId = '') {
  if (!State.courses.length) {
    return `<option value="">— no courses, add Canvas token in Settings —</option>`;
  }
  const sel = v => String(v) === String(selectedId) ? ' selected' : '';
  let html = includeAll ? `<option value="0"${sel('0')}>All courses</option>` : '';
  for (const c of State.courses) {
    html += `<option value="${c.id}"${sel(c.id)}>${esc(c.name)} (${c.id})</option>`;
  }
  return html;
}

function firstCourseId() {
  return State.courses.length ? String(State.courses[0].id) : '';
}

function courseNameFromId(id) {
  const c = State.courses.find(x => String(x.id) === String(id));
  return c ? c.name : `Course ${id}`;
}

function sectionTitle(text, iconHtml = '') {
  return `<div class="section-title">${iconHtml}${esc(text)}</div>`;
}

function mkCard(innerHTML, extra = '') {
  return `<div class="card ${extra}">${innerHTML}</div>`;
}

function mkField(label, inputHtml) {
  return `<div class="field"><label class="label">${esc(label)}</label>${inputHtml}</div>`;
}

function mkSwitch(id, label, checked = false) {
  return `<label class="switch-row">
    <span class="switch">
      <input type="checkbox" id="${id}" ${checked ? 'checked' : ''}>
      <span class="switch-track"></span>
      <span class="switch-thumb"></span>
    </span>
    <span>${esc(label)}</span>
  </label>`;
}

function mkCheckbox(id, label, checked = true) {
  return `<label class="checkbox-row">
    <input type="checkbox" id="${id}" ${checked ? 'checked' : ''}>
    <span>${esc(label)}</span>
  </label>`;
}

function mkRevealField(id, placeholder, value = '') {
  return `<div class="reveal-wrap">
    <input type="password" id="${id}" class="input-text"
      placeholder="${esc(placeholder)}" value="${esc(value)}">
    <button class="reveal-btn" onclick="toggleReveal('${id}')">👁</button>
  </div>`;
}

function toggleReveal(id) {
  const el = document.getElementById(id);
  if (el) el.type = el.type === 'password' ? 'text' : 'password';
}

// ── Run pipeline command ───────────────────────────────────────────────────────
function runCmd(cmd, label = '') {
  if (State.running) {
    Term.write('⚠  Already running — stop it first.', 'warn');
    return;
  }
  Term.write(`\n${'─'.repeat(60)}\n`, 'cmd');
  Term.showStop(true);
  Term.setStatus('● running…', 'var(--c-warn)');
  State.running = true;
  if (label) Term.write(`$ ${label}\n`, 'cmd');
  window.api.offProcessEvents();
  window.api.onProcessData(text => Term.process(text));
  window.api.onProcessDone(({ code }) => {
    State.running = false;
    Term.showStop(false);
    if (code === 0) {
      Term.write('\n✓  Completed successfully.', 'ok');
      Term.setStatus('✓ done', 'var(--c-success)');
    } else if (code === -15 || code === null) {
      // user stopped
    } else {
      Term.write(`\n✗  Exited with code ${code}.`, 'err');
      Term.setStatus(`✗ code ${code}`, 'var(--c-error)');
    }
  });
  window.api.runProcess(cmd);
}

// Pipeline chain: run steps in sequence, abort on failure
function runChain(steps) {
  if (!steps.length) return;
  if (State.running) {
    Term.write('⚠  Already running — stop it first.', 'warn');
    return;
  }
  let idx = 0;
  window.api.offProcessEvents();

  function runNext() {
    if (idx >= steps.length) {
      State.running = false;
      Term.showStop(false);
      Term.write('\n✓  All steps completed successfully.', 'ok');
      Term.setStatus('✓ done', 'var(--c-success)');
      return;
    }
    const [label, cmd] = steps[idx++];
    const stepLabel = `Step ${idx}/${steps.length}: ${label}`;
    Term.write(`\n${'─'.repeat(50)}\n▶  ${stepLabel}\n`, 'cmd');
    Term.showStop(true);
    Term.setStatus('● running…', 'var(--c-warn)');
    State.running = true;
    window.api.offProcessEvents();
    window.api.onProcessData(text => Term.process(text));
    window.api.onProcessDone(({ code }) => {
      if (code === 0) {
        runNext();
      } else if (code === -15 || code === null) {
        State.running = false;
        Term.showStop(false);
      } else {
        State.running = false;
        Term.showStop(false);
        Term.write(`\n✗  ${stepLabel} failed (code ${code}).`, 'err');
        Term.setStatus(`✗ code ${code}`, 'var(--c-error)');
      }
    });
    window.api.runProcess(cmd);
  }

  Term.write(`\n${'═'.repeat(60)}\n▶  Starting pipeline (${steps.length} step${steps.length !== 1 ? 's' : ''})\n`, 'cmd');
  runNext();
}

// ── Navigation ────────────────────────────────────────────────────────────────
const PAGES = [
  { label: 'Dashboard', icon: I.dashboard  },
  { label: 'Pipeline',  icon: I.pipeline   },
  { label: 'Download',  icon: I.download   },
  { label: 'Transcribe',icon: I.transcribe },
  { label: 'Align',     icon: I.align      },
  { label: 'Generate',  icon: I.generate   },
  { label: 'Settings',  icon: I.settings   },
];

const PAGE_BUILDERS = [
  buildDashboard,
  buildPipeline,
  buildDownload,
  buildTranscribe,
  buildAlign,
  buildGenerate,
  buildSettings,
];

function buildNav() {
  const container = document.getElementById('nav-items');
  container.innerHTML = PAGES.map((p, i) =>
    `<div class="nav-item${i === State.currentPage ? ' active' : ''}" data-idx="${i}" title="${p.label}">
      ${p.icon}
      <span class="nav-label">${p.label}</span>
    </div>`
  ).join('');
  container.querySelectorAll('.nav-item').forEach(el => {
    el.addEventListener('click', () => navigate(parseInt(el.dataset.idx)));
  });
}

function navigate(idx) {
  State.currentPage = idx;
  buildNav();
  renderPage();
}

function renderPage() {
  const el = document.getElementById('page-content');
  el.innerHTML = '';
  const content = PAGE_BUILDERS[State.currentPage]();
  if (typeof content === 'string') {
    el.innerHTML = content;
  } else {
    el.appendChild(content);
  }
  attachPageHandlers();
}

// ── Page: Dashboard ───────────────────────────────────────────────────────────
function buildDashboard() {
  if (!State.courses.length) {
    return `
      ${sectionTitle('Course Overview', I.dashboard + ' ')}
      ${mkCard(`<div class="empty-state">
        <svg width="52" height="52" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.2"><path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/><path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/></svg>
        <div class="empty-title">No courses loaded</div>
        <div class="empty-body">Go to Settings → enter your Canvas URL and API token,
        then save to load your courses automatically.</div>
        <button class="btn-secondary" onclick="navigate(6)">Open Settings</button>
      </div>`)}
      <div class="row">
        <button class="btn-icon" id="dash-refresh-btn">${I.refresh} Refresh Courses</button>
      </div>
    `;
  }

  let courseCards = '<div class="row" style="align-items:stretch">';
  for (const c of State.courses) {
    courseCards += `<div class="course-card" id="cc-${c.id}">
      <div class="course-header">
        <span class="course-short">${esc(c.name.split(' ')[0])}</span>
        <span class="chip chip-none">Loading…</span>
      </div>
      <div class="course-name">${esc(c.name)}</div>
      <hr class="divider">
      <div class="stat-row">
        <div class="stat-col"><span class="label">Videos</span><span>—</span></div>
        <div class="stat-col"><span class="label">Captions</span><span>—</span></div>
        <div class="stat-col"><span class="label">Aligned</span><span>—</span></div>
      </div>
      <div style="font-size:10px;color:var(--c-white-35);margin-top:4px">loading…</div>
    </div>`;
    if (State.courses.indexOf(c) % 2 === 1) courseCards += '</div><div class="row" style="align-items:stretch">';
  }
  courseCards += '</div>';

  const quickActions = `
    <div class="row" style="margin-top:4px">
      ${[
        ['Full Pipeline',  I.play, 1],
        ['Download',       I.download, 2],
        ['Transcribe',     I.transcribe, 3],
        ['Align',          I.align, 4],
        ['Generate Notes', I.generate, 5],
      ].map(([lbl, ico, idx]) =>
        `<button class="btn-icon" onclick="navigate(${idx})">${ico} ${lbl}</button>`
      ).join('')}
    </div>`;

  return `
    <div class="row">
      ${sectionTitle('Course Overview', '')}
      <div class="spacer"></div>
      <button class="btn-icon" id="dash-refresh-btn">${I.refresh} Refresh Courses</button>
    </div>
    ${courseCards}
    <div style="margin-top:8px">${sectionTitle('Quick Actions', '')}</div>
    ${quickActions}
  `;
}

async function loadDashboardStats() {
  for (const c of State.courses) {
    const el = document.getElementById(`cc-${c.id}`);
    if (!el) continue;
    try {
      const s = await window.api.getFilesStatus(c.id);
      const noteChip = s.notesFile
        ? chip('✓ notes', 'chip-success')
        : chip('○ pending', 'chip-none');
      el.querySelector('.course-header span:last-child').outerHTML = noteChip;
      el.querySelector('.stat-row').innerHTML = `
        <div class="stat-col"><span class="label">Videos</span>${statusChip(s.videos.done, s.videos.total)}</div>
        <div class="stat-col"><span class="label">Captions</span>${statusChip(s.captions, Math.max(s.captions, s.videos.done))}</div>
        <div class="stat-col"><span class="label">Aligned</span>${statusChip(s.alignments, Math.max(s.alignments, s.captions))}</div>
      `;
      el.querySelector('div:last-child').textContent = s.notesFile || 'no notes yet';
    } catch {}
  }
}

// ── Course detail modal (video list + delete) ────────────────────────────────
async function showCourseDetail(courseId) {
  const c = State.courses.find(x => String(x.id) === String(courseId));
  const label = c ? c.name : `Course ${courseId}`;

  // Create overlay
  const overlay = document.createElement('div');
  overlay.className = 'modal-overlay';
  overlay.innerHTML = `
    <div class="modal-panel">
      <div class="modal-header">
        <h2>${esc(label)}</h2>
        <button class="btn-icon" id="modal-close">${I.close || '&times;'} Close</button>
      </div>
      <div class="modal-body" id="modal-body">
        <div style="color:var(--c-white-45);padding:20px 0;text-align:center">Loading videos…</div>
      </div>
    </div>`;
  document.body.appendChild(overlay);

  // Close handlers
  overlay.querySelector('#modal-close').addEventListener('click', () => overlay.remove());
  overlay.addEventListener('click', e => { if (e.target === overlay) overlay.remove(); });

  // Load video data
  try {
    const videos = await window.api.listCourseVideos(courseId);
    const body = overlay.querySelector('#modal-body');

    if (!videos || videos.length === 0) {
      body.innerHTML = '<div style="color:var(--c-white-45);padding:20px 0;text-align:center">No transcribed videos found.<br>Run Transcribe first.</div>';
      return;
    }

    let html = '';
    for (const v of videos) {
      const alignChip = v.hasAlignment
        ? chip('aligned', 'chip-success')
        : chip('no align', 'chip-none');
      const notesChip = v.hasNotes
        ? chip('notes', 'chip-success')
        : chip('no notes', 'chip-none');
      html += `<div class="video-row" data-stem="${esc(v.stem)}">
        <div class="video-name">${esc(v.stem)}</div>
        <div class="video-chips">${alignChip}${notesChip}</div>
        <button class="btn-delete" data-stem="${esc(v.stem)}">Delete</button>
      </div>`;
    }
    body.innerHTML = html;

    // Delete handlers
    body.querySelectorAll('.btn-delete').forEach(btn => {
      btn.addEventListener('click', async (e) => {
        const stem = e.target.dataset.stem;
        if (!confirm(`Delete transcript, alignment, and notes for:\n${stem}?`)) return;
        try {
          const result = await window.api.deleteVideoData(courseId, stem);
          snack(`Deleted ${result.deleted.length} file(s)`);
          // Remove row from UI
          const row = body.querySelector(`.video-row[data-stem="${CSS.escape(stem)}"]`);
          if (row) row.remove();
          // Check if modal is now empty
          if (!body.querySelector('.video-row')) {
            body.innerHTML = '<div style="color:var(--c-white-45);padding:20px 0;text-align:center">All videos deleted.</div>';
          }
          // Refresh dashboard stats
          loadDashboardStats();
        } catch (err) {
          snack('Delete failed: ' + (err.message || err), false);
        }
      });
    });
  } catch (err) {
    const body = overlay.querySelector('#modal-body');
    body.innerHTML = `<div style="color:var(--c-error);padding:20px 0">Error loading videos: ${esc(err.message || String(err))}</div>`;
  }
}

// ── Page: Full Pipeline ────────────────────────────────────────────────────────
function buildPipeline() {
  const ps  = State.pipeline;
  const cid = ps.courseId || firstCourseId();
  return `
    ${sectionTitle('Full Pipeline Wizard', '')}
    ${mkCard(`
      <div class="row">
        <div class="col expand">
          <span class="label">Course</span>
          <select id="pp-course" class="select-ctrl">
            ${courseOptions(false, cid)}
          </select>
        </div>
      </div>
    `)}
    <div class="row" style="align-items:flex-start">
      ${mkCard(`
        <span class="label">Steps to execute</span>
        ${mkCheckbox('pp-step-mat',   'Download materials',   ps.steps.mat)}
        ${mkCheckbox('pp-step-vid',   'Download videos',      ps.steps.vid)}
        ${mkCheckbox('pp-step-trans', 'Transcribe videos',    ps.steps.trans)}
        ${mkCheckbox('pp-step-align', 'Align transcripts',    ps.steps.align)}
        ${mkCheckbox('pp-step-gen',   'Generate study notes', ps.steps.gen)}
      `, 'col expand')}
      ${mkCard(`
        <span class="label">Download</span>
        ${mkSwitch('pp-stealth', 'Slack mode for downloads', ps.stealth)}
        <hr class="divider">
        <span class="label">Note generation</span>
        <div class="row" style="gap:8px">
          <div class="field" style="flex:1">
            <label class="label">Course name for notes</label>
            <input id="pp-course-name" class="input-text" type="text" value="${esc(ps.courseName)}">
          </div>
          <div class="field" style="min-width:110px">
            <label class="label">Language</label>
            <select id="pp-language" class="select-ctrl">
              <option value="en"${ps.language === 'en' ? ' selected' : ''}>English</option>
              <option value="zh"${ps.language === 'zh' ? ' selected' : ''}>中文</option>
            </select>
          </div>
        </div>
        <div class="row center" style="gap:8px;margin-top:8px">
          <span class="slider-value" id="pp-detail-val">${esc(ps.detail)}</span>
          <div class="col expand">
            <input id="pp-detail" type="range" min="0" max="10" value="${esc(ps.detail)}" step="1" class="slider">
            <div style="font-size:10px;color:var(--c-white-35);margin-top:2px">
              0-2 Outline · 3-5 Bullets · 6-8 Paragraphs · 9-10 Exhaustive
            </div>
          </div>
        </div>
        <div class="field" style="margin-top:8px">
          <label class="label">Lecture filter</label>
          <input id="pp-lec-filter" class="input-text" type="text" value="${esc(ps.lecFilter)}" placeholder="1-5  or  1,3,5  (blank=all)">
        </div>
        ${mkSwitch('pp-force', 'Force regenerate', ps.force)}
      `, 'col expand')}
    </div>
    <div class="row">
      <button class="btn-primary" id="pp-run-btn">${I.play} Run Pipeline</button>
    </div>
  `;
}

// ── Page: Download ─────────────────────────────────────────────────────────────
function buildDownload() {
  return `
    ${sectionTitle('Download', '')}
    ${mkCard(`
      <div class="row">
        <div class="col expand">
          <span class="label">Course filter</span>
          <select id="dl-course" class="select-ctrl">
            ${courseOptions(true)}
          </select>
        </div>
        <div class="col">
          <span class="label">Options</span>
          ${mkSwitch('dl-stealth', 'Slack mode')}
        </div>
      </div>
    `)}
    ${mkCard(`
      <div class="row center" style="gap:8px">${I.video}<strong>Videos</strong></div>
      <div class="row" style="margin-top:8px">
        <button class="btn-outline" id="dl-vid-list">List videos</button>
        <button class="btn-outline" id="dl-vid-all">Download all pending</button>
      </div>
      <div class="row end" style="margin-top:8px">
        <div class="col expand">
          <input id="dl-vid-nums" class="input-text" type="text" placeholder="Video numbers, e.g. 1 3 5">
        </div>
        <button class="btn-primary" id="dl-vid-sel">${I.download} Download selected</button>
      </div>
    `)}
    ${mkCard(`
      <div class="row center" style="gap:8px">${I.folder}<strong>Course materials</strong></div>
      <div class="row" style="margin-top:8px">
        <button class="btn-outline" id="dl-mat-list">List materials</button>
        <button class="btn-outline" id="dl-mat-all">Download all pending</button>
      </div>
      <div class="row end" style="margin-top:8px">
        <div class="col expand">
          <input id="dl-mat-names" class="input-text" type="text" placeholder="Partial filename(s), space-separated">
        </div>
        <button class="btn-primary" id="dl-mat-sel">${I.download} Download selected</button>
      </div>
    `)}
  `;
}

// ── Page: Transcribe ───────────────────────────────────────────────────────────
function buildTranscribe() {
  const model = '__WHISPER_MODEL__';
  const lang  = '__WHISPER_LANG__';
  return `
    ${sectionTitle('Transcribe Videos', '')}
    <div id="trans-info-card"><!-- filled async --></div>
    ${mkCard(`
      <div class="field">
        <label class="label">Single video path (leave blank for all pending)</label>
        <input id="trans-path" class="input-text" type="text"
          placeholder="Leave blank to process all videos in manifest">
      </div>
      <div class="row" style="margin-top:12px">
        <button class="btn-primary" id="trans-run-btn">${I.play} Transcribe all pending</button>
      </div>
    `)}
  `;
}

async function fillTranscribeInfo() {
  const [model, lang] = await Promise.all([
    window.api.getConstant('transcribe', 'WHISPER_MODEL_SIZE'),
    window.api.getConstant('transcribe', 'WHISPER_LANGUAGE'),
  ]);
  const el = document.getElementById('trans-info-card');
  if (el) {
    el.innerHTML = mkCard(`<div class="info-row">${I.info}
      <span>Model: <strong>${esc(model)}</strong>   Language: <strong>${esc(lang === 'None' ? 'auto-detect' : lang)}</strong><br>
      Auto-selects backend: faster-whisper (GPU) or OpenAI Whisper API (CPU fallback, requires OpenAI key).</span></div>`);
  }
}

// ── Page: Align ────────────────────────────────────────────────────────────────

// Align page state
const AlignState = { rows: [], slideOptions: [], courseId: '' };

function buildAlign() {
  return `
    ${sectionTitle('Align Transcripts to Slides', '')}
    <div id="align-info-card"></div>
    ${mkCard(`
      <div class="card-title" style="display:flex;align-items:center;gap:8px">
        ${I.link} Video ↔ Slide Matching
      </div>
      <div class="card-sub">
        Match each video to its lecture slides. Auto-suggested matches are pre-filled.
        Click <strong>+</strong> to add multiple slide files, or <strong>×</strong> to exclude a video.
      </div>
      <div class="row" style="margin-top:8px;align-items:flex-end">
        <div class="col expand">
          <span class="label">Course</span>
          <select id="align-course" class="select-ctrl">${courseOptions()}</select>
        </div>
        <button class="btn-primary" id="align-scan-btn">${I.search || '🔍'} Scan videos &amp; slides</button>
      </div>
      <div class="row" style="margin-top:8px;align-items:flex-end;gap:8px">
        <div class="col">
          <span class="label">Matching model</span>
          <select id="align-match-model" class="select-ctrl" style="min-width:160px">
            <option value="bge-m3" selected>BGE-M3 (recommended)</option>
            <option value="jina">Jina Embeddings v4</option>
            <option value="mpnet">all-mpnet-base-v2</option>
          </select>
        </div>
        <button class="btn-outline" id="align-smart-btn">Smart match (embedding)</button>
      </div>
      <div id="align-match-status" style="font-size:11px;color:var(--c-white-45);margin:4px 0 6px 0"></div>
      <div id="align-match-rows"></div>
      <div class="row" style="margin-top:12px;gap:12px">
        <button class="btn-primary" id="align-run-btn">${I.play} Align with mapping</button>
        <button class="btn-outline" id="align-auto-btn">Auto-align (skip matching)</button>
      </div>
    `)}
  `;
}

function _alignSlideOptionsHtml(selected = '(none)') {
  let html = '<option value="(none)">(none — auto-detect)</option>';
  for (const s of AlignState.slideOptions) {
    const sel = s.rel === selected ? ' selected' : '';
    html += `<option value="${esc(s.rel)}"${sel}>${esc(s.rel)}</option>`;
  }
  return html;
}

// Auto-suggest is now computed server-side in main.js (align:scan handler)
// using lecture-number matching, date matching, transcript keyword overlap,
// and slide version preferences (with notes > review > plain).

function _alignRebuildRows() {
  const container = document.getElementById('align-match-rows');
  if (!container) return;

  if (!AlignState.rows.length) { container.innerHTML = ''; return; }

  let html = `<div style="display:flex;gap:6px;padding:2px 0;opacity:0.4;font-size:10px;font-weight:600">
    <span style="width:28px"></span><span style="width:14px"></span>
    <span style="width:220px">Video</span><span style="width:16px"></span>
    <span style="flex:1">Lecture slides</span></div>`;

  AlignState.rows.forEach((row, ri) => {
    const statusIcon = row.aligned
      ? '<span style="color:var(--c-success)" title="Already aligned">●</span>'
      : '<span style="opacity:0.3" title="Not yet aligned">○</span>';

    let slidesHtml = '';
    row.slides.forEach((sel, si) => {
      const actionBtn = si === 0
        ? `<button class="icon-btn" title="Add slide file" data-action="add" data-row="${ri}">+</button>`
        : `<button class="icon-btn" title="Remove" data-action="remove-slide" data-row="${ri}" data-slide="${si}" style="color:var(--c-error)">−</button>`;
      slidesHtml += `<div style="display:flex;align-items:center;gap:2px;margin-bottom:2px">
        <select class="select-ctrl" style="flex:1;font-size:11px" data-row="${ri}" data-slide="${si}">${_alignSlideOptionsHtml(sel)}</select>
        ${actionBtn}</div>`;
    });

    html += `<div style="display:flex;align-items:flex-start;gap:6px;padding:6px 0;border-bottom:1px solid rgba(255,255,255,0.06)">
      <button class="icon-btn" title="Remove video" data-action="remove" data-row="${ri}" style="color:var(--c-error);font-size:14px">×</button>
      <span style="width:14px;padding-top:4px">${statusIcon}</span>
      <span style="width:220px;padding-top:4px;font-size:11px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="${esc(row.title)}">${esc(row.title)}</span>
      <span style="padding-top:4px;opacity:0.3">→</span>
      <div style="flex:1">${slidesHtml}</div>
    </div>`;
  });

  container.innerHTML = html;

  // Update status
  const nMatched = AlignState.rows.filter(r => r.slides.some(s => s !== '(none)')).length;
  const statusEl = document.getElementById('align-match-status');
  if (statusEl) statusEl.textContent = `${AlignState.rows.length} video(s), ${AlignState.slideOptions.length} slide file(s). ${nMatched} matched.`;

  // Bind events via delegation
  container.onclick = (e) => {
    const btn = e.target.closest('[data-action]');
    if (!btn) return;
    const action = btn.dataset.action;
    const ri     = parseInt(btn.dataset.row);
    if (action === 'remove') {
      AlignState.rows.splice(ri, 1);
      _alignRebuildRows();
    } else if (action === 'add') {
      AlignState.rows[ri].slides.push('(none)');
      _alignRebuildRows();
    } else if (action === 'remove-slide') {
      const si = parseInt(btn.dataset.slide);
      if (AlignState.rows[ri].slides.length > 1) {
        AlignState.rows[ri].slides.splice(si, 1);
        _alignRebuildRows();
      }
    }
  };
  container.onchange = (e) => {
    const sel = e.target.closest('select[data-row]');
    if (!sel) return;
    const ri = parseInt(sel.dataset.row);
    const si = parseInt(sel.dataset.slide);
    AlignState.rows[ri].slides[si] = sel.value;
  };
}

async function fillAlignInfo() {
  const [model, ctx] = await Promise.all([
    window.api.getConstant('align', 'EMBED_MODEL'),
    window.api.getConstant('align', 'CONTEXT_SEC'),
  ]);
  const el = document.getElementById('align-info-card');
  if (el) {
    el.innerHTML = mkCard(`<div class="info-row">${I.info}
      <span>Embed: <strong>${esc(model)}</strong>   Context: ±<strong>${esc(ctx)}s</strong></span></div>`);
  }
}

// ── Page: Generate Notes ───────────────────────────────────────────────────────
function buildGenerate() {
  return `
    ${sectionTitle('Generate Study Notes', '')}
    <div id="gen-info-card"></div>
    <div class="row" style="align-items:flex-start">
      ${mkCard(`
        <span class="label">Course</span>
        <select id="gen-course" class="select-ctrl">${courseOptions()}</select>
        <div class="row" style="gap:8px;margin-top:8px">
          <div class="field" style="flex:1">
            <label class="label">Course name</label>
            <input id="gen-course-name" class="input-text" type="text">
          </div>
          <div class="field" style="min-width:110px">
            <label class="label">Language</label>
            <select id="gen-language" class="select-ctrl">
              <option value="en" selected>English</option>
              <option value="zh">中文</option>
            </select>
          </div>
        </div>
        <div class="field" style="margin-top:8px">
          <label class="label">Lecture</label>
          <select id="gen-lec-select" class="select-ctrl">
            <option value="">All lectures</option>
          </select>
          <div id="gen-lec-status" style="font-size:11px;color:var(--c-white-45);margin-top:4px;min-height:14px"></div>
        </div>
      `, 'col expand')}
      ${mkCard(`
        <span class="label">Detail level</span>
        <div class="row center" style="gap:8px;margin-top:6px">
          <span class="slider-value" id="gen-detail-val">7</span>
          <div class="col expand">
            <input id="gen-detail" type="range" min="0" max="10" value="7" step="1" class="slider">
            <div style="font-size:10px;color:var(--c-white-35);margin-top:2px">
              0-2 Outline · 3-5 Bullets · 6-8 Paragraphs · 9-10 Exhaustive
            </div>
          </div>
        </div>
        <hr class="divider">
        <span class="label">Options</span>
        ${mkSwitch('gen-force',   'Force regenerate all sections')}
        ${mkSwitch('gen-merge',   'Merge-only (skip generation)')}
        ${mkSwitch('gen-iterate', 'Iterative mode (raise detail until quality target)')}
      `, 'col expand')}
    </div>
    <div class="row">
      <button class="btn-primary" id="gen-run-btn">✦ Generate Notes</button>
    </div>
  `;
}

// Populate the lecture dropdown for a given course id
async function loadLectureDropdown(cid) {
  const sel    = document.getElementById('gen-lec-select');
  const status = document.getElementById('gen-lec-status');
  if (!sel) return;
  sel.innerHTML = '<option value="">Loading…</option>';
  sel.disabled = true;
  if (status) status.textContent = '';

  const lecs = await window.api.listLectures(cid);
  sel.disabled = false;

  if (!lecs || lecs.length === 0) {
    sel.innerHTML = '<option value="">All lectures (no slides found)</option>';
    if (status) status.textContent = 'No slide files found under materials/';
    return;
  }

  // Only show lectures with extracted numbers; tutorials/midterms etc. have no number
  const numbered = lecs.filter(l => l.num != null);
  let opts = `<option value="">All lectures (${numbered.length})</option>`;
  for (const l of numbered) {
    opts += `<option value="${l.num}">L${l.num}: ${l.title}</option>`;
  }
  sel.innerHTML = opts;

  const alignCount = lecs[0]?.alignCount ?? 0;
  const alignHint  = alignCount > 0 ? `  ·  ${alignCount} alignment file(s) ready` : '  ·  no alignment files yet';
  if (status) status.textContent = `${numbered.length} lecture(s) found${alignHint}`;

  // Update run button label when selection changes
  sel.addEventListener('change', () => {
    const btn = document.getElementById('gen-run-btn');
    if (!btn) return;
    btn.textContent = sel.value ? `✦ Generate Notes for L${sel.value}` : '✦ Generate Notes';
  });
}

async function fillGenInfo() {
  const [noteModel, verifyModel, target] = await Promise.all([
    window.api.getConstant('generate', 'NOTE_MODEL'),
    window.api.getConstant('generate', 'VERIFY_MODEL'),
    window.api.getConstant('generate', 'QUALITY_TARGET'),
  ]);
  const el = document.getElementById('gen-info-card');
  if (el) {
    el.innerHTML = mkCard(`<div class="info-row">${I.info}
      <span>Generator: <strong>${esc(noteModel)}</strong>   Verifier: <strong>${esc(verifyModel)}</strong>
        Quality target: <strong>${esc(target)}</strong></span></div>`);
  }
}

// ── Page: Settings ─────────────────────────────────────────────────────────────
function buildSettings() {
  return `
    ${sectionTitle('Settings', '')}
    ${buildSettingsConnection()}
    ${buildSettingsKeys()}
    ${buildSettingsEnv()}
    ${buildSettingsConstants()}
    <div class="row" style="margin-top:8px;margin-bottom:16px">
      <button class="btn-primary" id="settings-save-btn">💾 Save All Settings</button>
      <button class="btn-secondary" id="settings-refresh-btn">${I.refresh} Refresh Courses</button>
      <span id="settings-refresh-status" style="font-size:11px;color:var(--c-white-60)"></span>
    </div>
    ${buildUninstallSection()}
  `;
}

function buildUninstallSection() {
  return `
    <div style="margin-top:24px;padding:16px;border:1px solid var(--c-red,#e53935);border-radius:8px;background:rgba(229,57,53,0.06)">
      <div style="font-weight:600;font-size:14px;color:var(--c-red,#e53935);margin-bottom:6px">Danger Zone — Uninstall AutoNote</div>
      <div style="font-size:12px;color:var(--c-white-60);margin-bottom:12px">
        Choose what to remove. Unchecked items will be permanently deleted.
      </div>
      <div id="uninstall-sizes" style="font-size:12px;color:var(--c-white-60);margin-bottom:12px;line-height:1.8">
        Calculating sizes…
      </div>
      <div style="display:flex;flex-direction:column;gap:8px;margin-bottom:14px">
        <label style="display:flex;align-items:center;gap:8px;font-size:13px;cursor:pointer">
          <input type="checkbox" id="uninstall-keep-venv" checked>
          Keep ML environment (PyTorch, Whisper, sentence-transformers, …)
        </label>
        <label style="display:flex;align-items:center;gap:8px;font-size:13px;cursor:pointer">
          <input type="checkbox" id="uninstall-keep-content" checked>
          Keep generated content (notes, captions, alignment, videos, materials)
        </label>
        <label style="display:flex;align-items:center;gap:8px;font-size:13px;cursor:pointer">
          <input type="checkbox" id="uninstall-keep-settings" checked>
          Keep settings and config (Canvas token, API keys, output directory)
        </label>
      </div>
      <button id="btn-uninstall" style="background:var(--c-red,#e53935);color:#fff;border:none;padding:8px 18px;border-radius:6px;font-size:13px;cursor:pointer;font-weight:600">
        Uninstall AutoNote
      </button>
    </div>
  `;
}

function buildSettingsConnection() {
  return mkCard(`
    <div class="card-title">Connection</div>
    <div class="card-sub">Saved to config.json in the app data directory.</div>
    <div id="conn-fields"><!-- loaded async --></div>
  `);
}

function buildSettingsKeys() {
  return mkCard(`
    <div class="card-title">API Keys &amp; Credentials</div>
    <div class="card-sub">Keys are stored in plaintext files in ~/.auto_note/</div>
    <div id="keys-fields"><!-- loaded async --></div>
    <div style="font-size:11px;color:var(--c-white-45);margin-top:8px">
      Data dir: <span id="data-dir-display" style="color:var(--c-white-60);user-select:text"></span>
    </div>
  `);
}

function buildSettingsEnv() {
  return mkCard(`
    <div class="card-title">ML Environment</div>
    <div class="card-sub">Creates ~/.auto_note/venv/ — select which components to install.
      Without local ML, the app uses online APIs (requires API keys).</div>
    <div class="row center" style="gap:8px;margin-bottom:8px">
      <span style="font-size:11px;color:var(--c-white-45)">Status:</span>
      <span id="env-status" style="font-size:11px"></span>
    </div>
    <div id="env-components" style="margin-bottom:10px"><!-- filled async --></div>
    <div class="row" style="margin-bottom:8px;gap:8px">
      <button class="btn-secondary" id="env-install-btn">⬇ Install ML Environment</button>
      <button class="btn-outline" id="env-reinstall-btn" style="display:none">↺ Reinstall</button>
    </div>
    <div id="env-manual-py-row" style="display:none;margin-bottom:8px">
      <input id="env-manual-py" class="input-text" type="text"
        placeholder="Python path (auto-detected — paste here only if auto-detect fails)">
    </div>
    <textarea id="env-log" class="install-log" readonly style="display:none"></textarea>
  `);
}

async function fillEnvComponents() {
  const comps = await window.api.getInstallComponents();
  const el = document.getElementById('env-components');
  if (!el || !comps) return;
  let html = '';
  for (const [key, def] of Object.entries(comps)) {
    const checked = 'checked';
    const disabled = def.required ? 'disabled' : '';
    const note = def.note ? ` <span style="color:var(--c-white-35);font-size:10px">(${def.note})</span>` : '';
    html += `<label style="display:flex;align-items:center;gap:6px;padding:2px 0;cursor:pointer;font-size:12px;color:var(--c-white-70)">
      <input type="checkbox" class="env-comp-cb" data-comp="${esc(key)}" ${checked} ${disabled}
        style="accent-color:var(--c-primary)">
      ${esc(def.label)}${note}
    </label>`;
  }
  el.innerHTML = html;
}

const CONSTANTS_DEF = [
  // [scriptKey, name, desc, default, options | null]
  ['transcribe', 'WHISPER_MODEL_SIZE', 'Whisper model variant', 'large-v3', [
    'tiny','base','small','medium','large','large-v2','large-v3','large-v3-turbo','distil-large-v3'
  ]],
  ['transcribe', 'WHISPER_LANGUAGE', 'Transcription language', 'None', [
    ['Auto-detect','None'],['English','en'],['Chinese','zh'],['Japanese','ja'],
    ['Korean','ko'],['French','fr'],['German','de'],['Spanish','es'],
  ]],
  ['align', 'EMBED_MODEL', 'Sentence-transformer model', 'all-mpnet-base-v2', [
    ['all-mpnet-base-v2 (best quality)','all-mpnet-base-v2'],
    ['all-MiniLM-L12-v2 (balanced)','all-MiniLM-L12-v2'],
    ['all-MiniLM-L6-v2 (fast)','all-MiniLM-L6-v2'],
    ['paraphrase-multilingual-mpnet-base-v2','paraphrase-multilingual-mpnet-base-v2'],
  ]],
  ['align', 'CONTEXT_SEC',        'Context window (s)',       '30',   null],
  ['align', 'OFF_SLIDE_THRESHOLD','Off-slide cosine cutoff',  '0.28', null],
  ['align', 'PRIOR_SIGMA',        'Temporal prior σ',         '5',    null],
  ['generate', 'NOTE_MODEL', 'Note generation LLM', 'gpt-5.1', [
    // ── OpenAI ──────────────────────────────────────────────────────────
    ['GPT-5.1 ★','gpt-5.1'],['GPT-5.2','gpt-5.2'],
    ['GPT-4.1','gpt-4.1'],['GPT-4.1 mini','gpt-4.1-mini'],['GPT-4.1 nano','gpt-4.1-nano'],
    ['o3 (reasoning)','o3'],['o4-mini (reasoning)','o4-mini'],['o3-mini','o3-mini'],
    // ── Anthropic ───────────────────────────────────────────────────────
    ['Claude Opus 4.6','claude-opus-4-6'],['Claude Sonnet 4.6','claude-sonnet-4-6'],
    ['Claude Sonnet 4.5','claude-sonnet-4-5'],['Claude Haiku 4.5','claude-haiku-4-5-20251001'],
    ['Claude Sonnet 3.5','claude-3-5-sonnet-20241022'],['Claude Haiku 3.5','claude-3-5-haiku-20241022'],
    // ── Google Gemini ───────────────────────────────────────────────────
    ['Gemini 2.5 Pro','gemini-2.5-pro'],['Gemini 2.5 Flash','gemini-2.5-flash'],
    ['Gemini 2.5 Flash Lite','gemini-2.5-flash-lite'],['Gemini 2.0 Flash','gemini-2.0-flash'],
    // ── DeepSeek ────────────────────────────────────────────────────────
    ['DeepSeek V3 (chat)','deepseek-chat'],['DeepSeek R1 (reasoning)','deepseek-reasoner'],
    // ── xAI Grok ────────────────────────────────────────────────────────
    ['Grok 3','grok-3'],['Grok 3 mini','grok-3-mini'],
    // ── Mistral ─────────────────────────────────────────────────────────
    ['Mistral Large','mistral-large-latest'],['Mistral Medium','mistral-medium-latest'],
    ['Mistral Small','mistral-small-latest'],['Codestral','codestral-latest'],
  ]],
  ['generate', 'VERIFY_MODEL', 'Verification LLM', 'gpt-4.1-mini', [
    // ── OpenAI ──────────────────────────────────────────────────────────
    ['GPT-4.1 mini ★','gpt-4.1-mini'],['GPT-4.1 nano','gpt-4.1-nano'],
    ['GPT-4.1','gpt-4.1'],['GPT-5.1','gpt-5.1'],['o4-mini (reasoning)','o4-mini'],
    // ── Anthropic ───────────────────────────────────────────────────────
    ['Claude Haiku 4.5','claude-haiku-4-5-20251001'],['Claude Haiku 3.5','claude-3-5-haiku-20241022'],
    ['Claude Sonnet 4.6','claude-sonnet-4-6'],['Claude Sonnet 3.5','claude-3-5-sonnet-20241022'],
    // ── Google Gemini ───────────────────────────────────────────────────
    ['Gemini 2.5 Flash','gemini-2.5-flash'],['Gemini 2.5 Flash Lite','gemini-2.5-flash-lite'],
    ['Gemini 2.0 Flash','gemini-2.0-flash'],
    // ── DeepSeek ────────────────────────────────────────────────────────
    ['DeepSeek V3 (chat)','deepseek-chat'],
    // ── xAI / Mistral ───────────────────────────────────────────────────
    ['Grok 3 mini','grok-3-mini'],['Mistral Small','mistral-small-latest'],
  ]],
  ['generate', 'DETAIL_LEVEL',   'Default detail level',  '8',   null],
  ['generate', 'CHAPTER_SIZE',   'Slides per GPT call',   '15',  null],
  ['generate', 'QUALITY_TARGET', 'Self-score target',     '8.0', null],
];

const SCRIPT_COLORS = {
  transcribe: { fg: '#4DD0E1', bg: 'rgba(77,208,225,0.12)' },
  align:      { fg: '#FFD54F', bg: 'rgba(255,213,79,0.12)' },
  generate:   { fg: '#CE93D8', bg: 'rgba(206,147,216,0.12)' },
};

function buildSettingsConstants() {
  let rows = CONSTANTS_DEF.map(([key, name, desc, def, opts], i) => {
    const { fg, bg } = SCRIPT_COLORS[key] || { fg: 'var(--c-primary)', bg: 'transparent' };
    let ctrl;
    if (opts) {
      const optHtml = opts.map(o => {
        const [label, val] = Array.isArray(o) ? o : [o, o];
        return `<option value="${esc(val)}">${esc(label)}</option>`;
      }).join('');
      ctrl = `<select class="select-ctrl const-ctrl" data-const-idx="${i}" id="const-${i}">${optHtml}</select>`;
    } else {
      ctrl = `<input type="text" class="input-text const-ctrl" data-const-idx="${i}" id="const-${i}" value="${esc(def)}">`;
    }
    return `<div class="const-row">
      <span class="const-script-badge" style="color:${fg};background:${bg}">${key}</span>
      <span class="const-desc">${esc(desc)}</span>
      <span class="const-name">${esc(name)}</span>
      <div class="const-ctrl">${ctrl}</div>
      <button class="btn-outline" style="padding:4px 8px;font-size:11px"
        data-const-reset="${i}" title="Reset to default: ${esc(def)}">Default</button>
    </div>`;
  }).join('');

  return mkCard(`
    <div class="card-title">Tunable Constants</div>
    <div class="card-sub">Changes are written directly to the corresponding script file.</div>
    <hr class="divider">
    <div id="const-rows">
      ${rows}
    </div>
  `);
}

// ── Settings async data loading ────────────────────────────────────────────────
async function loadSettingsData() {
  const [cfg, creds, venv, dataDir] = await Promise.all([
    window.api.getConfig(),
    window.api.getCredentials(),
    window.api.getVenvStatus(),
    window.api.getDataDir(),
  ]);

  // Data dir display
  const dd = document.getElementById('data-dir-display');
  if (dd) dd.textContent = dataDir || '~/.auto_note';

  // Connection fields
  const connEl = document.getElementById('conn-fields');
  if (connEl) {
    connEl.innerHTML = `
      <div class="row">
        <div class="col expand">
          <label class="label">Canvas URL</label>
          <input id="cfg-canvas-url" class="input-text" type="text"
            value="${esc(cfg.CANVAS_URL || '')}"
            placeholder="canvas.yourschool.edu  (https:// added automatically)">
        </div>
      </div>
      <div class="row">
        <div class="col expand">
          <label class="label">Panopto Host</label>
          <input id="cfg-panopto" class="input-text" type="text"
            value="${esc(cfg.PANOPTO_HOST || '')}"
            placeholder="mediaweb.ap.panopto.com">
        </div>
        <div class="col expand">
          <label class="label">Python Path</label>
          <input id="cfg-python" class="input-text" type="text"
            value="${esc(cfg.PYTHON_PATH || '')}"
            placeholder="blank = auto-detect (uses ~/.auto_note/venv)">
        </div>
      </div>
      <div class="row">
        <div class="col expand">
          <label class="label">Output Directory</label>
          <div class="row end">
            <input id="cfg-output-dir" class="input-text" type="text"
              value="${esc(cfg.OUTPUT_DIR || '')}"
              placeholder="~/AutoNote" style="flex:1">
            <button class="btn-outline" id="cfg-browse-btn" style="white-space:nowrap">Browse…</button>
          </div>
        </div>
      </div>
      <div class="row">
        <div class="col expand">
          <label class="label">Transcription Backend</label>
          <select id="cfg-whisper-backend" class="input-text" style="flex:1">
            <option value="auto"  ${(cfg.WHISPER_BACKEND||'auto')==='auto' ?'selected':''}>Auto (GPU if ≥16 GB VRAM, else API)</option>
            <option value="gpu"   ${(cfg.WHISPER_BACKEND||'auto')==='gpu'  ?'selected':''}>Force GPU (faster-whisper)</option>
            <option value="api"   ${(cfg.WHISPER_BACKEND||'auto')==='api'  ?'selected':''}>Force API (OpenAI Whisper)</option>
          </select>
        </div>
      </div>
    `;
  }

  // API Keys fields
  const keysEl = document.getElementById('keys-fields');
  if (keysEl) {
    keysEl.innerHTML = `
      <div class="row">
        <div class="col expand"><label class="label">Canvas Token</label>
          ${mkRevealField('cred-canvas', 'Canvas API token', creds.canvas)}</div>
      </div>
      <div class="row">
        <div class="col expand"><label class="label">OpenAI API Key</label>
          ${mkRevealField('cred-openai', 'sk-…', creds.openai)}</div>
        <div class="col expand"><label class="label">Anthropic API Key</label>
          ${mkRevealField('cred-anthropic', 'sk-ant-…', creds.anthropic)}</div>
      </div>
      <div class="row">
        <div class="col expand"><label class="label">Gemini API Key</label>
          ${mkRevealField('cred-gemini', 'AIza…', creds.gemini)}</div>
        <div class="col expand"><label class="label">DeepSeek API Key</label>
          ${mkRevealField('cred-deepseek', 'sk-…', creds.deepseek)}</div>
      </div>
      <div class="row">
        <div class="col expand"><label class="label">xAI (Grok) API Key</label>
          ${mkRevealField('cred-grok', 'xai-…', creds.grok)}</div>
        <div class="col expand"><label class="label">Mistral API Key</label>
          ${mkRevealField('cred-mistral', 'key…', creds.mistral)}</div>
      </div>
    `;
  }

  // Venv status
  updateVenvStatus(venv);

  // Load constant values
  await loadConstantValues();
}

function updateVenvStatus(venv) {
  const statusEl    = document.getElementById('env-status');
  const installBtn  = document.getElementById('env-install-btn');
  const reinstallBtn = document.getElementById('env-reinstall-btn');
  if (statusEl) {
    if (venv.installed) {
      statusEl.textContent = `✓ Installed at ${venv.path}`;
      statusEl.style.color = 'var(--c-success)';
    } else {
      statusEl.textContent = 'Not installed';
      statusEl.style.color = 'var(--c-white-45)';
    }
  }
  if (reinstallBtn) reinstallBtn.style.display = venv.installed ? 'inline-flex' : 'none';
}

async function loadConstantValues() {
  for (let i = 0; i < CONSTANTS_DEF.length; i++) {
    const [key, name, , def] = CONSTANTS_DEF[i];
    const el = document.getElementById(`const-${i}`);
    if (!el) continue;
    try {
      const val = await window.api.getConstant(key, name);
      el.value = val === '?' ? def : val;
    } catch {
      el.value = def;
    }
  }
}

async function saveAllSettings() {
  const btn = document.getElementById('settings-save-btn');
  if (btn) btn.disabled = true;

  const errors = [];

  // Config
  try {
    const cfgData = {
      CANVAS_URL:       (document.getElementById('cfg-canvas-url')?.value || '').trim(),
      PANOPTO_HOST:     (document.getElementById('cfg-panopto')?.value || '').trim(),
      PYTHON_PATH:      (document.getElementById('cfg-python')?.value || '').trim(),
      OUTPUT_DIR:       (document.getElementById('cfg-output-dir')?.value || '').trim(),
      WHISPER_BACKEND:  document.getElementById('cfg-whisper-backend')?.value || 'auto',
    };
    await window.api.setConfig(cfgData);
    // Update runtime output dir
    State.outputDir = cfgData.OUTPUT_DIR;
  } catch (e) { errors.push('Config: ' + e.message); }

  // Credentials
  try {
    await window.api.setCredentials({
      canvas:    document.getElementById('cred-canvas')?.value || '',
      openai:    document.getElementById('cred-openai')?.value || '',
      anthropic: document.getElementById('cred-anthropic')?.value || '',
      gemini:    document.getElementById('cred-gemini')?.value || '',
      deepseek:  document.getElementById('cred-deepseek')?.value || '',
      grok:      document.getElementById('cred-grok')?.value || '',
      mistral:   document.getElementById('cred-mistral')?.value || '',
    });
  } catch (e) { errors.push('Credentials: ' + e.message); }

  // Constants
  for (let i = 0; i < CONSTANTS_DEF.length; i++) {
    const [key, name, , def] = CONSTANTS_DEF[i];
    const el = document.getElementById(`const-${i}`);
    if (!el) continue;
    const val = el.value.trim() || def;
    try {
      const ok = await window.api.setConstant(key, name, val);
      if (!ok) errors.push(`Failed to write ${name}`);
    } catch (e) { errors.push(`${name}: ${e.message}`); }
  }

  if (btn) btn.disabled = false;

  if (errors.length) {
    snack('Errors: ' + errors.join('; '), false);
  } else {
    snack('All settings saved successfully!');
    // Auto-refresh courses
    refreshCourses(true);
  }
}

// ── Refresh courses ────────────────────────────────────────────────────────────
async function refreshCourses(fromSettings = false) {
  const statusEl = fromSettings
    ? document.getElementById('settings-refresh-status')
    : null;

  if (statusEl) statusEl.textContent = 'Refreshing…';

  const result = await window.api.fetchCourses();
  if (result.error) {
    if (statusEl) {
      statusEl.textContent = '✗ ' + result.error;
      statusEl.style.color = 'var(--c-error)';
    } else {
      snack('Course refresh failed: ' + result.error, false);
    }
    return;
  }

  State.courses = result.courses || [];
  const n = State.courses.length;

  if (statusEl) {
    statusEl.textContent = n ? `✓ ${n} course${n !== 1 ? 's' : ''} loaded.` : 'No courses found.';
    statusEl.style.color = n ? 'var(--c-success)' : 'var(--c-warn)';
  }

  // Rebuild nav and current page to reflect new courses
  if (State.courses.length && State.currentPage === 6) {
    navigate(0);
  } else {
    renderPage();
  }
}

// ── Page handler attachment (called after renderPage) ─────────────────────────
async function attachPageHandlers() {
  const pg = State.currentPage;

  // ── Dashboard ────────────────────────────────────────────────────────────────
  if (pg === 0) {
    document.getElementById('dash-refresh-btn')?.addEventListener('click', () => refreshCourses());
    // Click on course card → show detail modal with video list
    for (const c of State.courses) {
      const card = document.getElementById(`cc-${c.id}`);
      if (card) card.addEventListener('click', () => showCourseDetail(c.id));
    }
    loadDashboardStats();
    return;
  }

  // ── Pipeline ─────────────────────────────────────────────────────────────────
  if (pg === 1) {
    const detSlider = document.getElementById('pp-detail');
    const detLabel  = document.getElementById('pp-detail-val');
    detSlider?.addEventListener('input', () => {
      detLabel.textContent = detSlider.value;
      State.pipeline.detail = detSlider.value;
    });

    // Persist all form values so they survive navigation
    document.getElementById('pp-course')?.addEventListener('change', e => {
      State.pipeline.courseId = e.target.value;
      const nameEl = document.getElementById('pp-course-name');
      if (nameEl) {
        nameEl.value = courseNameFromId(e.target.value);
        State.pipeline.courseName = nameEl.value;
      }
    });
    document.getElementById('pp-course-name')?.addEventListener('input', e => {
      State.pipeline.courseName = e.target.value;
    });
    document.getElementById('pp-lec-filter')?.addEventListener('input', e => {
      State.pipeline.lecFilter = e.target.value;
    });
    document.getElementById('pp-language')?.addEventListener('change', e => {
      State.pipeline.language = e.target.value;
    });
    document.getElementById('pp-stealth')?.addEventListener('change', e => {
      State.pipeline.stealth = e.target.checked;
    });
    document.getElementById('pp-force')?.addEventListener('change', e => {
      State.pipeline.force = e.target.checked;
    });
    for (const [k, id] of [
      ['mat','pp-step-mat'],['vid','pp-step-vid'],['trans','pp-step-trans'],
      ['align','pp-step-align'],['gen','pp-step-gen'],
    ]) {
      document.getElementById(id)?.addEventListener('change', e => {
        State.pipeline.steps[k] = e.target.checked;
      });
    }

    // Auto-fill course name if blank on first render
    const ppCourseEl = document.getElementById('pp-course');
    const ppNameEl   = document.getElementById('pp-course-name');
    if (ppCourseEl && ppNameEl && !ppNameEl.value.trim() && ppCourseEl.value) {
      ppNameEl.value = courseNameFromId(ppCourseEl.value);
      State.pipeline.courseName = ppNameEl.value;
    }

    document.getElementById('pp-run-btn')?.addEventListener('click', async () => {
      const cid     = document.getElementById('pp-course')?.value;
      if (!cid) { snack('Select a course first.', false); return; }
      const python  = await window.api.getPythonPath();
      const outDir  = State.outputDir || await window.api.getOutputDir();
      const stealth = document.getElementById('pp-stealth')?.checked;
      const force   = document.getElementById('pp-force')?.checked;
      const name    = document.getElementById('pp-course-name')?.value.trim() || '';
      const detail  = document.getElementById('pp-detail')?.value || '7';
      const lang    = document.getElementById('pp-language')?.value || 'en';
      const lf      = document.getElementById('pp-lec-filter')?.value.trim() || '';
      const steps = [
        ['dl_mat',   document.getElementById('pp-step-mat')?.checked],
        ['dl_vid',   document.getElementById('pp-step-vid')?.checked],
        ['transcribe', document.getElementById('pp-step-trans')?.checked],
        ['align',    document.getElementById('pp-step-align')?.checked],
        ['generate', document.getElementById('pp-step-gen')?.checked],
      ].filter(([, v]) => v).map(([k]) => k);

      if (!steps.length) { snack('No steps selected.', false); return; }

      const paths = await window.api.getScriptsPaths();
      const chain = [];

      if (steps.includes('dl_mat')) {
        const c = [python, paths.downloader, '--course', cid, '--download-material-all', '--path', outDir];
        if (stealth) c.push('--secretly');
        chain.push(['Download materials', c]);
      }
      if (steps.includes('dl_vid')) {
        const c = [python, paths.downloader, '--course', cid, '--download-video-all', '--path', outDir];
        if (stealth) c.push('--secretly');
        chain.push(['Download videos', c]);
      }
      if (steps.includes('transcribe')) {
        const c = [python, paths.transcribe];
        if (force) c.push('--force');
        chain.push(['Transcribe', c]);
      }
      if (steps.includes('align')) {
        const c = [python, paths.align, '--course', cid];
        if (force) c.push('--force');
        chain.push(['Align', c]);
      }
      if (steps.includes('generate')) {
        const c = [python, paths.generate, '--course', cid, '--course-name', name || courseNameFromId(cid), '--detail', detail, '--language', lang, '--per-video'];
        if (lf) c.push('--lectures', lf);
        if (force) c.push('--force');
        chain.push(['Generate notes', c]);
      }

      runChain(chain);
    });
    return;
  }

  // ── Download ──────────────────────────────────────────────────────────────────
  if (pg === 2) {
    const getArgs = async () => {
      const python  = await window.api.getPythonPath();
      const paths   = await window.api.getScriptsPaths();
      const outDir  = State.outputDir || await window.api.getOutputDir();
      const cid     = document.getElementById('dl-course')?.value || '0';
      const stealth = document.getElementById('dl-stealth')?.checked;
      const base = [python, paths.downloader, '--path', outDir];
      if (cid !== '0') base.push('--course', cid);
      if (stealth)     base.push('--secretly');
      return { base, paths, outDir, python };
    };

    document.getElementById('dl-vid-list')?.addEventListener('click', async () => {
      const { base } = await getArgs(); runCmd([...base, '--video-list'], '--video-list');
    });
    document.getElementById('dl-vid-all')?.addEventListener('click', async () => {
      const { base } = await getArgs();
      runCmd([...base, '--download-video-all', '--transcribe'], '--download-video-all --transcribe');
    });
    document.getElementById('dl-vid-sel')?.addEventListener('click', async () => {
      const nums = document.getElementById('dl-vid-nums')?.value.trim().split(/\s+/).filter(Boolean);
      if (!nums?.length) { snack('Enter video number(s) first.', false); return; }
      const { base } = await getArgs();
      runCmd([...base, '--download-video', ...nums, '--transcribe'], '--download-video --transcribe');
    });
    document.getElementById('dl-mat-list')?.addEventListener('click', async () => {
      const { base } = await getArgs(); runCmd([...base, '--material-list'], '--material-list');
    });
    document.getElementById('dl-mat-all')?.addEventListener('click', async () => {
      const { base } = await getArgs(); runCmd([...base, '--download-material-all'], '--download-material-all');
    });
    document.getElementById('dl-mat-sel')?.addEventListener('click', async () => {
      const names = document.getElementById('dl-mat-names')?.value.trim().split(/\s+/).filter(Boolean);
      if (!names?.length) { snack('Enter filename(s) first.', false); return; }
      const { base } = await getArgs(); runCmd([...base, '--download-material', ...names], '--download-material');
    });
    return;
  }

  // ── Transcribe ────────────────────────────────────────────────────────────────
  if (pg === 3) {
    fillTranscribeInfo();
    document.getElementById('trans-run-btn')?.addEventListener('click', async () => {
      const python = await window.api.getPythonPath();
      const paths  = await window.api.getScriptsPaths();
      const vid    = document.getElementById('trans-path')?.value.trim();
      const cmd = [python, paths.transcribe];
      if (vid) cmd.push('--video', vid);
      runCmd(cmd, 'extract_caption.py');
    });
    return;
  }

  // ── Align ─────────────────────────────────────────────────────────────────────
  if (pg === 4) {
    fillAlignInfo();

    // Scan button — populate the matching rows
    document.getElementById('align-scan-btn')?.addEventListener('click', async () => {
      const cid = document.getElementById('align-course')?.value;
      if (!cid) { snack('Select a course first.', false); return; }
      AlignState.courseId = cid;

      const data = await window.api.alignScan(cid);
      AlignState.slideOptions = data.slides;

      AlignState.rows = data.captions.map(cap => {
        // Determine title
        const title = data.titles[cap.stem] || cap.stem;
        // Pick initial slides: existing mapping > auto-suggest
        let initSlides;
        if (data.mapping[cap.stem] && data.mapping[cap.stem].length) {
          initSlides = data.mapping[cap.stem];
        } else {
          // Use server-side suggestion (computed with transcript keywords + date matching)
          initSlides = cap.suggested ? [cap.suggested] : ['(none)'];
        }
        return { stem: cap.stem, title, aligned: cap.aligned, transcribed: cap.transcribed, slides: initSlides };
      });

      _alignRebuildRows();
      snack(`Found ${data.captions.length} video(s), ${data.slides.length} slide file(s).`);
    });

    // Smart match button — runs as a terminal command (reliable, visible output)
    document.getElementById('align-smart-btn')?.addEventListener('click', async () => {
      const cid = AlignState.courseId || document.getElementById('align-course')?.value;
      if (!cid) { snack('Select a course first.', false); return; }
      if (!AlignState.rows.length) { snack('Click "Scan" first.', false); return; }

      const model = document.getElementById('align-match-model')?.value || 'bge-m3';
      const statusEl = document.getElementById('align-match-status');
      if (statusEl) statusEl.textContent = `Running ${model} smart match…`;

      const python = await window.api.getPythonPath();
      const paths  = await window.api.getScriptsPaths();
      const cmd = [python, paths.align, '--course', cid,
                   '--suggest-matches', '--match-model', model];

      // Run as a visible terminal command — user sees all output
      if (State.running) { snack('Already running — stop it first.', false); return; }
      Term.write(`\n${'─'.repeat(60)}\n`, 'cmd');
      Term.showStop(true);
      Term.setStatus('● matching…', 'var(--c-warn)');
      State.running = true;
      Term.write(`$ smart-match --model ${model}\n`, 'cmd');

      window.api.offProcessEvents();
      window.api.onProcessData(text => Term.process(text));
      window.api.onProcessDone(async ({ code }) => {
        State.running = false;
        Term.showStop(false);
        if (code === 0) {
          Term.write('\n✓ Smart match completed.', 'ok');
          Term.setStatus('✓ done', 'var(--c-success)');
          // Reload the mapping file that the script saved
          const data = await window.api.alignScan(cid);
          const mapKeys = data.mapping ? Object.keys(data.mapping) : [];
          const rowStems = AlignState.rows.map(r => r.stem);
          Term.write(`\n── Reload mapping ──\nMapping keys: ${JSON.stringify(mapKeys)}\nRow stems: ${JSON.stringify(rowStems)}\n`, 'cmd');
          if (mapKeys.length) {
            let updated = 0;
            for (const row of AlignState.rows) {
              if (data.mapping[row.stem] && data.mapping[row.stem].length) {
                row.slides = data.mapping[row.stem];
                updated++;
              }
            }
            _alignRebuildRows();
            snack(`Smart match applied: ${updated} video(s) matched.`);
          } else {
            Term.write('Mapping file empty or not found by align:scan.\n', 'warn');
            snack('Smart match saved no matches — check terminal output.', false);
          }
        } else if (code === -15 || code === null) {
          // user stopped
        } else {
          Term.write(`\n✗  Smart match failed (code ${code}).`, 'err');
          Term.setStatus(`✗ code ${code}`, 'var(--c-error)');
        }
        if (statusEl) statusEl.textContent = '';
      });
      window.api.runProcess(cmd);
    });

    // Align with mapping button
    document.getElementById('align-run-btn')?.addEventListener('click', async () => {
      const cid = AlignState.courseId || document.getElementById('align-course')?.value;
      if (!cid) { snack('Select a course first.', false); return; }
      if (!AlignState.rows.length) { snack('Click "Scan" first.', false); return; }

      // Build mapping from current state
      const mapping = {};
      for (const row of AlignState.rows) {
        const vals = row.slides.filter(s => s && s !== '(none)');
        if (vals.length) mapping[row.stem] = vals;
      }

      // Save mapping
      const mapFile = await window.api.alignSaveMapping(cid, mapping);

      const python = await window.api.getPythonPath();
      const paths  = await window.api.getScriptsPaths();
      const cmd = [python, paths.align, '--course', cid, '--mapping', mapFile];
      runCmd(cmd, 'align --course ' + cid + ' --mapping …');
    });

    // Auto-align (skip matching) button
    document.getElementById('align-auto-btn')?.addEventListener('click', async () => {
      const cid = document.getElementById('align-course')?.value;
      if (!cid) { snack('Select a course first.', false); return; }
      const python = await window.api.getPythonPath();
      const paths  = await window.api.getScriptsPaths();
      const cmd = [python, paths.align, '--course', cid];
      runCmd(cmd, 'align --course ' + cid);
    });
    return;
  }

  // ── Generate ──────────────────────────────────────────────────────────────────
  if (pg === 5) {
    fillGenInfo();
    const detSlider = document.getElementById('gen-detail');
    const detLabel  = document.getElementById('gen-detail-val');
    detSlider?.addEventListener('input', () => { detLabel.textContent = detSlider.value; });

    // Auto-fill course name + reload lecture list when course changes
    const genCourseEl = document.getElementById('gen-course');
    genCourseEl?.addEventListener('change', (e) => {
      const nameEl = document.getElementById('gen-course-name');
      if (nameEl) nameEl.value = courseNameFromId(e.target.value);
      loadLectureDropdown(e.target.value);
    });

    // Pre-fill course name and load lectures for default course
    const firstCid = genCourseEl?.value;
    if (firstCid) {
      const nameEl = document.getElementById('gen-course-name');
      if (nameEl && !nameEl.value) nameEl.value = courseNameFromId(firstCid);
      loadLectureDropdown(firstCid);
    }

    document.getElementById('gen-run-btn')?.addEventListener('click', async () => {
      const cid    = document.getElementById('gen-course')?.value;
      if (!cid) { snack('Select a course first.', false); return; }

      const python  = await window.api.getPythonPath();
      const paths   = await window.api.getScriptsPaths();
      const name    = document.getElementById('gen-course-name')?.value.trim() || courseNameFromId(cid);
      const detail  = document.getElementById('gen-detail')?.value || '7';
      const lang    = document.getElementById('gen-language')?.value || 'en';
      const force   = document.getElementById('gen-force')?.checked;
      const merge   = document.getElementById('gen-merge')?.checked;
      const iter    = document.getElementById('gen-iterate')?.checked;
      const lecNum  = document.getElementById('gen-lec-select')?.value;   // '' = all

      const cmd = [python, paths.generate, '--course', cid, '--course-name', name, '--detail', detail, '--language', lang];
      if (lecNum) cmd.push('--lectures', lecNum);
      if (force)  cmd.push('--force');
      if (merge)  cmd.push('--merge-only');
      if (iter)   cmd.push('--iterate');

      const label = lecNum
        ? `note_generation.py  L${lecNum} of course ${cid}`
        : `note_generation.py --course ${cid}`;
      runCmd(cmd, label);
    });
    return;
  }

  // ── Settings ──────────────────────────────────────────────────────────────────
  if (pg === 6) {
    loadSettingsData();
    fillEnvComponents();

    document.getElementById('cfg-browse-btn')?.addEventListener('click', async () => {
      const dir = await window.api.openDirDialog();
      if (dir) {
        const el = document.getElementById('cfg-output-dir');
        if (el) el.value = dir;
      }
    });

    document.getElementById('settings-save-btn')?.addEventListener('click', saveAllSettings);

    document.getElementById('settings-refresh-btn')?.addEventListener('click', () => {
      refreshCourses(true);
    });

    // Uninstall sizes
    (async () => {
      const el = document.getElementById('uninstall-sizes');
      if (!el) return;
      try {
        const s = await window.api.getUninstallSizes();
        const fmt = mb => {
          if (mb >= 1024) return `${(mb / 1024).toFixed(1)} GB`;
          if (mb >= 1)    return `${mb} MB`;
          return '< 1 MB';
        };
        const venvLabel = s.venv === 0 ? '<span style="color:var(--c-white-45)">not installed</span>' : `<b>${fmt(s.venv)}</b>`;
        let html = `<b>Disk usage (controlled by checkboxes below):</b><br>`;
        html += `&nbsp;&nbsp;• ML Environment (~/.auto_note/venv): ${venvLabel}<br>`;
        html += `&nbsp;&nbsp;• Settings &amp; config (~/.auto_note/): <b>${fmt(s.settings)}</b><br>`;
        if (s.content !== null) {
          html += `&nbsp;&nbsp;• Generated content (${s.outputDir}): <b>${fmt(s.content)}</b>`;
        } else {
          html += `&nbsp;&nbsp;• Generated content: <span style="color:var(--c-white-45)">inside ~/.auto_note (removed with settings)</span>`;
        }
        el.innerHTML = html;
      } catch { el.textContent = 'Could not calculate sizes.'; }
    })();

    // Uninstall button
    document.getElementById('btn-uninstall')?.addEventListener('click', async () => {
      const keepVenv     = document.getElementById('uninstall-keep-venv')?.checked ?? true;
      const keepContent  = document.getElementById('uninstall-keep-content')?.checked ?? true;
      const keepSettings = document.getElementById('uninstall-keep-settings')?.checked ?? true;
      const lines = ['This will permanently delete:'];
      if (!keepVenv)     lines.push('  • ML environment (~/.auto_note/venv)');
      if (!keepSettings) lines.push('  • All settings, API keys, and credentials');
      if (!keepContent)  lines.push('  • All generated content (notes, captions, videos…)');
      if (lines.length === 1) lines.push('  (nothing — all items are set to Keep)');
      lines.push('', 'This cannot be undone. Continue?');

      if (!confirm(lines.join('\n'))) return;

      const btn = document.getElementById('btn-uninstall');
      btn.disabled = true;
      btn.textContent = 'Uninstalling…';

      const result = await window.api.runUninstall(keepContent, keepSettings, keepVenv);
      if (result.ok) {
        const isWin = result.platform === 'win32';
        alert(
          'AutoNote data has been removed.\n\n' +
          (isWin
            ? 'The Windows uninstaller has been launched — follow the prompts to remove the app from your system.'
            : 'To finish, delete the AutoNote AppImage file.')
        );
        window.close();
      } else {
        btn.disabled = false;
        btn.textContent = 'Uninstall AutoNote';
        alert(`Uninstall failed: ${result.error}`);
      }
    });

    // Venv install / reinstall
    const startInstall = async () => {
      const logEl   = document.getElementById('env-log');
      const btnInst = document.getElementById('env-install-btn');
      const btnRe   = document.getElementById('env-reinstall-btn');
      const statusEl = document.getElementById('env-status');
      if (logEl)   { logEl.value = ''; logEl.style.display = 'block'; }
      if (btnInst) btnInst.disabled = true;
      if (btnRe)   btnRe.disabled   = true;
      if (statusEl){ statusEl.textContent = 'Setting up…'; statusEl.style.color = 'var(--c-warn)'; }

      const manualPy = document.getElementById('env-manual-py')?.value.trim() || null;

      window.api.offInstallEvents();
      window.api.onInstallData(text => {
        if (logEl) { logEl.value += text; logEl.scrollTop = logEl.scrollHeight; }
      });
      window.api.onInstallDone(({ success, needsPath, error }) => {
        if (btnInst) btnInst.disabled = false;
        if (btnRe)   btnRe.disabled   = false;
        if (success) {
          if (statusEl) { statusEl.textContent = '✓ Installed'; statusEl.style.color = 'var(--c-success)'; }
          if (btnRe) btnRe.style.display = 'inline-flex';
          snack('ML environment installed successfully!');
        } else if (needsPath) {
          const row = document.getElementById('env-manual-py-row');
          if (row) row.style.display = 'block';
          if (statusEl) { statusEl.textContent = '✗ Python not found — enter path below'; statusEl.style.color = 'var(--c-error)'; }
        } else {
          if (statusEl) { statusEl.textContent = '✗ Setup failed'; statusEl.style.color = 'var(--c-error)'; }
          snack('Installation failed: ' + (error || 'unknown error'), false);
        }
      });

      // Gather selected components from checkboxes
      const compCbs = document.querySelectorAll('.env-comp-cb');
      const selComps = [];
      compCbs.forEach(cb => { if (cb.checked) selComps.push(cb.dataset.comp); });

      await window.api.startInstall(manualPy, selComps.length ? selComps : null);
    };

    document.getElementById('env-install-btn')?.addEventListener('click', startInstall);
    document.getElementById('env-reinstall-btn')?.addEventListener('click', startInstall);

    // Constants reset buttons
    document.querySelectorAll('[data-const-reset]').forEach(btn => {
      btn.addEventListener('click', async () => {
        const i   = parseInt(btn.dataset.constReset);
        const [key, name, , def] = CONSTANTS_DEF[i];
        const el  = document.getElementById(`const-${i}`);
        if (el) el.value = def;
        const ok = await window.api.setConstant(key, name, def);
        snack(ok ? `${name} reset to default.` : `Failed to reset ${name}.`, ok);
      });
    });
  }
}

// ── Terminal stop/clear buttons ────────────────────────────────────────────────
document.getElementById('term-stop')?.addEventListener('click', () => {
  window.api.stopProcess();
  Term.write('\n[stopped by user]', 'warn');
  Term.showStop(false);
  Term.setStatus('■ stopped', 'var(--c-warn)');
  State.running = false;
});

document.getElementById('term-clear')?.addEventListener('click', () => {
  Term.clear();
});

// ── Initial load ───────────────────────────────────────────────────────────────
async function init() {
  // Load output dir once so it's available for command building
  try {
    State.outputDir = await window.api.getOutputDir();
  } catch {}

  // Fetch courses
  const result = await window.api.fetchCourses().catch(() => ({ courses: [] }));
  State.courses = result?.courses || [];

  // If no courses loaded (not configured yet), start on Settings so the user
  // can enter their Canvas URL/token without having to click through the empty Dashboard.
  if (State.courses.length === 0) {
    State.currentPage = 6;
  }

  // Build UI
  buildNav();
  renderPage();
}

init().catch(err => {
  // If init() throws (e.g. IPC timeout or JS error), show a visible error
  // instead of a silent blank screen.
  document.body.innerHTML =
    `<div style="padding:40px;font-family:monospace;color:#EF5350;background:#1E2A2A;height:100vh">
      <h2 style="margin-bottom:16px">AutoNote failed to start</h2>
      <pre style="white-space:pre-wrap;font-size:12px">${String(err?.stack || err)}</pre>
      <p style="margin-top:16px;color:#aaa;font-size:12px">
        Press F12 to open DevTools for more details, then reload (Ctrl+R).
      </p>
    </div>`;
});
