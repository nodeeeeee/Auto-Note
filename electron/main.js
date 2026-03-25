'use strict';

const { app, BrowserWindow, ipcMain, dialog } = require('electron');
const path   = require('path');
const os     = require('os');
const fs     = require('fs');
const https  = require('https');
const http   = require('http');
const { spawnSync, spawn } = require('child_process');

// ── Optional: node-pty for real tty on Unix (tqdm in-place refresh) ──────────
let nodePty = null;
try { nodePty = require('node-pty'); } catch { /* fallback to pipe mode */ }

// ── Path constants ────────────────────────────────────────────────────────────
const DATA_DIR    = path.join(os.homedir(), '.auto_note');
const VENV_PYTHON = process.platform === 'win32'
  ? path.join(DATA_DIR, 'venv', 'Scripts', 'python.exe')
  : path.join(DATA_DIR, 'venv', 'bin', 'python');

// When packaged (asar), extraResources land in process.resourcesPath/scripts/.
// In dev mode (no asar), __dirname is electron/ and scripts live one level up.
const PROJECT_DIR = path.join(__dirname, '..');
const SCRIPTS_DIR_INSTALLED = path.join(DATA_DIR, 'scripts');

function scriptPath(name) {
  // 1. Packaged resources always win when running as AppImage/exe/dmg
  //    (ensures bundled scripts are never shadowed by stale installed copies)
  if (process.resourcesPath) {
    const packed = path.join(process.resourcesPath, 'scripts', name);
    if (fs.existsSync(packed)) return packed;
  }
  // 2. Dev mode: sibling of electron/
  return path.join(PROJECT_DIR, name);
}

const SCRIPTS = {
  downloader: scriptPath('downloader.py'),
  transcribe: scriptPath('extract_caption.py'),
  align:      scriptPath('semantic_alignment.py'),
  generate:   scriptPath('note_generation.py'),
};

const ML_PACKAGES = [
  'tqdm', 'faster-whisper', 'sentence-transformers', 'faiss-cpu',
  'pymupdf', 'python-pptx', 'python-docx', 'openai', 'anthropic',
  'google-generativeai', 'requests', 'pillow', 'httpx', 'playwright',
  'canvasapi', 'ffmpeg-progress-yield', 'pycryptodomex',
  // yarl installed separately (modern wheel) before PanoptoDownloader
  'yarl',
];
// PanoptoDownloader requires yarl~=1.7.2 in its metadata, but works fine with
// modern yarl at runtime.  Installing with --no-deps avoids the C++ build for
// the old yarl on Windows where no pre-built wheel exists for Python 3.12+.
//
// Use the zip archive URL instead of git+https://, because:
//   • On Windows: avoids triggering a git.exe subprocess (no PATH issues)
//   • On Linux AppImage: avoids system git, which fails when the AppImage's
//     bundled libssl.so.3 pollutes LD_LIBRARY_PATH and breaks libcurl
// pip downloads the zip using its own HTTP client (Python ssl), not libcurl.
const PANOPTO_PKG = 'https://github.com/Panopto-Video-DL/Panopto-Video-DL-lib/archive/refs/tags/v1.4.3.zip';

// ── Config helpers ────────────────────────────────────────────────────────────
function ensureDataDir() {
  fs.mkdirSync(DATA_DIR, { recursive: true });
}

function loadConfig() {
  const f = path.join(DATA_DIR, 'config.json');
  if (fs.existsSync(f)) {
    try { return JSON.parse(fs.readFileSync(f, 'utf8')); } catch {}
  }
  return {};
}

function saveConfig(data) {
  ensureDataDir();
  const f   = path.join(DATA_DIR, 'config.json');
  const cfg = loadConfig();
  Object.assign(cfg, data);
  fs.writeFileSync(f, JSON.stringify(cfg, null, 2));
}

function getOutputDir() {
  const cfg = loadConfig();
  return (cfg.OUTPUT_DIR || '').trim() || path.join(os.homedir(), 'AutoNote');
}

function getPythonPath() {
  if (fs.existsSync(VENV_PYTHON)) return VENV_PYTHON;
  const cfg = loadConfig();
  if (cfg.PYTHON_PATH && fs.existsSync(cfg.PYTHON_PATH)) return cfg.PYTHON_PATH;
  if (process.platform === 'win32') {
    const w = findExecutable('python');
    return w || 'python';
  }
  return findExecutable('python3') || findExecutable('python') || 'python3';
}

function findExecutable(name) {
  const cmd = process.platform === 'win32' ? 'where' : 'which';
  const r   = spawnSync(cmd, [name], { encoding: 'utf8', timeout: 3000, windowsHide: true });
  if (r.status === 0 && r.stdout) return r.stdout.trim().split('\n')[0].trim();
  return null;
}

// ── Credentials ───────────────────────────────────────────────────────────────
function loadCredentials() {
  const read = (f) => {
    const p = path.join(DATA_DIR, f);
    return fs.existsSync(p) ? fs.readFileSync(p, 'utf8').trim() : '';
  };
  return {
    canvas:    read('canvas_token.txt'),
    openai:    read('openai_api.txt'),
    anthropic: read('anthropic_key.txt'),
    gemini:    read('gemini_api.txt'),
  };
}

function saveCredentials(data) {
  ensureDataDir();
  const map = {
    canvas:    'canvas_token.txt',
    openai:    'openai_api.txt',
    anthropic: 'anthropic_key.txt',
    gemini:    'gemini_api.txt',
  };
  for (const [key, file] of Object.entries(map)) {
    if (data[key] !== undefined && data[key].trim()) {
      fs.writeFileSync(path.join(DATA_DIR, file), data[key].trim());
    }
  }
}

// ── Canvas API ────────────────────────────────────────────────────────────────
function httpGet(url, headers) {
  return new Promise((resolve, reject) => {
    const mod  = url.startsWith('https') ? https : http;
    const req  = mod.get(url, { headers }, (res) => {
      let body = '';
      res.on('data', d => body += d);
      res.on('end', () => {
        if (res.statusCode === 401) {
          return reject(new Error('401 Unauthorized — Canvas token is invalid or expired.'));
        }
        if (res.statusCode >= 400) {
          return reject(new Error(`HTTP ${res.statusCode}: ${url}`));
        }
        try { resolve(JSON.parse(body)); }
        catch { reject(new Error('Invalid JSON response from server')); }
      });
    });
    req.on('error', reject);
    req.setTimeout(12000, () => { req.destroy(); reject(new Error('Request timed out')); });
  });
}

const SKIP_KW = [
  'training', 'pdp', 'rmcpdp', 'osa', 'soct', 'travel',
  'essentials', 'respect', 'consent', 'osh',
];

async function fetchCoursesFromCanvas() {
  const tokenFile = path.join(DATA_DIR, 'canvas_token.txt');
  const token = fs.existsSync(tokenFile) ? fs.readFileSync(tokenFile, 'utf8').trim() : '';
  if (!token) return { error: 'Canvas token not saved — enter it in Settings → API Keys.' };
  const cfg = loadConfig();
  let url = (cfg.CANVAS_URL || '').trim().replace(/\/$/, '');
  if (!url) return { error: 'Canvas URL not configured — enter it in Settings → Connection.' };
  if (!url.startsWith('http')) url = 'https://' + url;
  try {
    const data = await httpGet(
      `${url}/api/v1/courses?enrollment_state=active&per_page=100`,
      { 'Authorization': `Bearer ${token}` },
    );
    const courses = data
      .filter(c => c.name && !SKIP_KW.some(k => c.name.toLowerCase().includes(k)))
      .map(c => ({ id: c.id, name: c.name || c.course_code || String(c.id) }));
    return { courses };
  } catch (err) {
    return { error: String(err.message || err) };
  }
}

// ── File status (Dashboard) ───────────────────────────────────────────────────
function getFilesStatus(courseId) {
  const outDir   = getOutputDir();
  const courseDir = path.join(outDir, String(courseId));

  // Videos: count mp4 files actually on disk (manifest can have stale/duplicate entries)
  const videosDir = path.join(courseDir, 'videos');
  let videoDone = 0, videoTotal = 0;
  if (fs.existsSync(videosDir)) {
    const mp4s = fs.readdirSync(videosDir).filter(f => f.toLowerCase().endsWith('.mp4'));
    videoTotal = mp4s.length;
    videoDone  = mp4s.length;  // present on disk = downloaded
  }

  const countJson = (dir, excludePattern) => {
    if (!fs.existsSync(dir)) return 0;
    return fs.readdirSync(dir).filter(f => f.endsWith('.json') && (!excludePattern || !f.includes(excludePattern))).length;
  };

  const notesDir = path.join(courseDir, 'notes');
  let notesFile = null, courseName = null;
  if (fs.existsSync(notesDir)) {
    const mds = fs.readdirSync(notesDir).filter(f => f.endsWith('.md') && !f.endsWith('.score.json'));
    if (mds.length > 0) {
      notesFile  = mds[0];
      courseName = notesFile.replace(/_notes\.md$/, '').replace(/_/g, ' ');
    }
  }

  return {
    videos:     { done: videoDone, total: videoTotal },
    captions:   countJson(path.join(courseDir, 'captions')),
    alignments: countJson(path.join(courseDir, 'alignment'), 'compact'),
    notesFile,
    courseName,
  };
}

// ── Script constants ──────────────────────────────────────────────────────────
function readConstant(scriptKey, name) {
  const sp = SCRIPTS[scriptKey];
  if (!sp || !fs.existsSync(sp)) return '?';
  try {
    const src = fs.readFileSync(sp, 'utf8');
    const m   = src.match(new RegExp(`^${name}\\s*=\\s*(.+)`, 'm'));
    if (!m) return '?';
    return m[1].replace(/#.*$/, '').trim().replace(/^["']|["']$/g, '');
  } catch { return '?'; }
}

function writeConstant(scriptKey, name, value) {
  const sp = SCRIPTS[scriptKey];
  if (!sp || !fs.existsSync(sp)) return false;
  try {
    const src = fs.readFileSync(sp, 'utf8');
    let newVal;
    if (value === 'None') {
      newVal = 'None';
    } else if (value !== '' && !isNaN(Number(value))) {
      newVal = value;  // bare numeric
    } else {
      newVal = `"${value}"`;  // quoted string
    }
    const re     = new RegExp(`^(${name}\\s*=\\s*)([^\\n#]+)`, 'm');
    const newSrc = src.replace(re, `$1${newVal}`);
    if (newSrc === src) return false;
    fs.writeFileSync(sp, newSrc);
    return true;
  } catch { return false; }
}

// ── Python detection for venv installer ───────────────────────────────────────
async function findSystemPython() {
  if (process.platform !== 'win32') {
    for (const shell of ['bash', 'zsh']) {
      try {
        const result = await new Promise((resolve, reject) => {
          const proc = spawn(shell, ['-l', '-c',
            "python3 -c 'import ssl,venv,sys; print(sys.executable)'"],
            { timeout: 15000, windowsHide: true });
          let out = '';
          proc.stdout?.on('data', d => out += d);
          proc.on('close', code => code === 0 ? resolve(out) : reject());
          proc.on('error', reject);
        });
        for (const line of result.trim().split('\n').reverse()) {
          const p = line.trim();
          if (p && fs.existsSync(p)) return p;
        }
      } catch {}
    }
  }

  const home = os.homedir();
  const candidates = process.platform === 'win32' ? [
    path.join(home, 'miniconda3',  'python.exe'),
    path.join(home, 'Miniconda3',  'python.exe'),
    path.join(home, 'anaconda3',   'python.exe'),
    path.join(home, 'Anaconda3',   'python.exe'),
    path.join(home, 'miniforge3',  'python.exe'),
    path.join(home, 'mambaforge',  'python.exe'),
    path.join(home, 'AppData', 'Local', 'Programs', 'Python', 'Python313', 'python.exe'),
    path.join(home, 'AppData', 'Local', 'Programs', 'Python', 'Python312', 'python.exe'),
    path.join(home, 'AppData', 'Local', 'Programs', 'Python', 'Python311', 'python.exe'),
    path.join(home, 'AppData', 'Local', 'Programs', 'Python', 'Python310', 'python.exe'),
    'C:\\miniconda3\\python.exe',
    'C:\\anaconda3\\python.exe',
    'C:\\ProgramData\\miniconda3\\python.exe',
    'C:\\ProgramData\\anaconda3\\python.exe',
  ] : [
    path.join(home, 'miniconda3',  'bin', 'python3'),
    path.join(home, 'miniconda3',  'bin', 'python'),
    path.join(home, 'anaconda3',   'bin', 'python3'),
    path.join(home, 'anaconda3',   'bin', 'python'),
    path.join(home, 'miniforge3',  'bin', 'python3'),
    path.join(home, 'miniforge3',  'bin', 'python'),
    path.join(home, 'mambaforge',  'bin', 'python3'),
    path.join(home, '.local', 'share', 'mamba', 'bin', 'python3'),
    '/opt/conda/bin/python3',
    '/opt/miniconda3/bin/python3',
    '/opt/anaconda3/bin/python3',
    '/usr/bin/python3',
    '/usr/local/bin/python3',
  ];

  for (const cand of candidates) {
    if (!cand || !fs.existsSync(cand)) continue;
    const r = spawnSync(cand, ['-c', 'import ssl, venv'], { timeout: 5000, windowsHide: true });
    if (r.status === 0) return cand;
  }
  return null;
}

function detectCuda() {
  try {
    const r = spawnSync('nvidia-smi', [], { encoding: 'utf8', timeout: 10000, windowsHide: true });
    if (r.status === 0) {
      const m = r.stdout.match(/CUDA Version:\s*(\d+)\.(\d+)/);
      if (m) return [parseInt(m[1]), parseInt(m[2])];
      return [12, 0];
    }
  } catch {}
  return null;
}

function torchIndexUrl(cuda) {
  if (!cuda) return null;
  const [major, minor] = cuda;
  const v = major * 10 + minor;
  if (v >= 128) return 'https://download.pytorch.org/whl/cu128';
  if (v >= 126) return 'https://download.pytorch.org/whl/cu126';
  if (v >= 124) return 'https://download.pytorch.org/whl/cu124';
  if (v >= 121) return 'https://download.pytorch.org/whl/cu121';
  if (v >= 118) return 'https://download.pytorch.org/whl/cu118';
  // CUDA < 11.8: too old for torch 2.x; fall back to CPU build
  return null;
}

// ── Active process state ──────────────────────────────────────────────────────
let activeProc    = null;
let installProc   = null;
let mainWindow    = null;
let _userStopped  = false;  // set true when user clicks Stop, so close code is ignored

// ── Subprocess runner (pipeline scripts) ─────────────────────────────────────
function runProcess(cmd) {
  if (activeProc) return { error: 'Already running — stop it first.' };
  _userStopped = false;

  const outDir = getOutputDir();
  fs.mkdirSync(outDir, { recursive: true });
  const env = {
    ...process.env,
    PYTHONUNBUFFERED:          '1',
    PYTHONIOENCODING:          'utf-8',  // prevent UnicodeEncodeError on Windows cp1252 consoles
    PYTHONUTF8:                '1',      // Python 3.7+ UTF-8 mode (also forces utf-8 on Windows)
    AUTONOTE_DATA_DIR:         DATA_DIR, // tell scripts where to find config/credentials
    AUTONOTE_WHISPER_BACKEND:  loadConfig().WHISPER_BACKEND || 'auto',
  };

  // Unix: prefer node-pty so tqdm sees a real tty and uses \r refresh
  if (process.platform !== 'win32' && nodePty) {
    try {
      const ptyProc = nodePty.spawn(cmd[0], cmd.slice(1), {
        name: 'xterm-256color',
        cols: 120,
        rows: 40,
        cwd: outDir,
        env: { ...env, TERM: 'xterm-256color', COLUMNS: '120' },
      });
      activeProc = ptyProc;
      ptyProc.onData(data => {
        mainWindow?.webContents.send('process:data', data);
      });
      ptyProc.onExit(({ exitCode }) => {
        activeProc = null;
        if (_userStopped) {
          mainWindow?.webContents.send('process:done', { code: null });
        } else {
          mainWindow?.webContents.send('process:done', { code: exitCode });
        }
      });
      return { ok: true };
    } catch { /* fall through to pipe mode */ }
  }

  // Pipe mode (Windows or pty unavailable)
  const [prog, ...args] = cmd;
  const proc = spawn(prog, args, {
    cwd: outDir,
    env,
    stdio: ['ignore', 'pipe', 'pipe'],
    windowsHide: true,
  });
  activeProc = proc;
  proc.stdout.on('data', d => mainWindow?.webContents.send('process:data', d.toString('utf8')));
  proc.stderr.on('data', d => mainWindow?.webContents.send('process:data', d.toString('utf8')));
  proc.on('close', code => {
    activeProc = null;
    if (_userStopped) {
      // User clicked Stop — don't treat as an error regardless of exit code
      mainWindow?.webContents.send('process:done', { code: null });
    } else {
      mainWindow?.webContents.send('process:done', { code: code ?? 1 });
    }
  });
  proc.on('error', err => {
    if (_userStopped) { activeProc = null; return; }
    activeProc = null;
    const hint = err.code === 'ENOENT'
      ? `\n[error] Could not find executable: ${prog}\n[error] Install the ML environment from Settings → ML Environment.`
      : '';
    mainWindow?.webContents.send('process:data', `[error] ${err.message}${hint}\n`);
    mainWindow?.webContents.send('process:done', { code: 1 });
  });
  return { ok: true };
}

function stopProcess() {
  if (!activeProc) return;
  _userStopped = true;
  try {
    if (typeof activeProc.kill === 'function') activeProc.kill();
    else activeProc.kill('SIGTERM');
  } catch {}
  activeProc = null;
  // Send 'stopped' event immediately so the renderer doesn't wait for close
  mainWindow?.webContents.send('process:done', { code: null });
}

// ── ML environment installer ──────────────────────────────────────────────────
async function runInstaller(basePython, sendLog, sendDone) {
  const venvDir = path.join(DATA_DIR, 'venv');

  // Build a clean environment for installer subprocesses.
  // On Linux AppImage, LD_LIBRARY_PATH is prepended with the AppImage's bundled
  // libs (e.g. libssl.so.3). System tools like git use system libcurl which then
  // finds the wrong libssl → OPENSSL version mismatch. Restore original value.
  const installerEnv = { ...process.env, PYTHONUNBUFFERED: '1', PYTHONIOENCODING: 'utf-8', PYTHONUTF8: '1' };
  if (process.platform === 'linux' && process.env.APPIMAGE) {
    const orig = process.env.APPIMAGE_ORIG_LD_LIBRARY_PATH;
    if (orig !== undefined) {
      installerEnv.LD_LIBRARY_PATH = orig;
    } else {
      delete installerEnv.LD_LIBRARY_PATH;
    }
  }

  const runStep = (cmd, args) => new Promise((resolve) => {
    const proc = spawn(cmd, args, {
      stdio: ['ignore', 'pipe', 'pipe'],
      env: installerEnv,
      windowsHide: true,
    });
    installProc = proc;
    proc.stdout.on('data', d => sendLog(d.toString('utf8')));
    proc.stderr.on('data', d => sendLog(d.toString('utf8')));
    proc.on('close', code => { installProc = null; resolve(code); });
    proc.on('error', err => { installProc = null; sendLog(`ERROR: ${err.message}\n`); resolve(-1); });
  });

  if (!basePython) {
    sendLog('► Probing system for Python 3 with SSL support…\n');
    basePython = await findSystemPython();
  }

  if (!basePython) {
    sendLog('ERROR: Could not find a Python 3 with SSL support.\n');
    if (process.platform === 'win32') {
      sendLog('  Paste your Python path in the field below (e.g. C:\\Users\\you\\miniconda3\\python.exe)\n');
    } else {
      sendLog('  Paste your Python path in the field below (e.g. /home/user/miniconda3/bin/python3)\n');
    }
    sendLog('  then click Install again.\n');
    sendDone({ success: false, needsPath: true });
    return;
  }

  sendLog(`► Using Python: ${basePython}\n`);

  // Step 1 — create venv (remove stale one first)
  if (fs.existsSync(venvDir)) {
    sendLog('► Removing old venv…\n');
    fs.rmSync(venvDir, { recursive: true, force: true });
  }
  sendLog(`► Creating venv at ${venvDir}…\n`);
  const venvCode = await runStep(basePython, ['-m', 'venv', venvDir]);
  if (venvCode !== 0) {
    sendLog('ERROR: venv creation failed.\n');
    sendDone({ success: false, error: 'venv creation failed' });
    return;
  }
  sendLog('  venv created.\n');

  const pip = process.platform === 'win32'
    ? path.join(venvDir, 'Scripts', 'pip.exe')
    : path.join(venvDir, 'bin', 'pip');

  // Step 2 — upgrade pip
  sendLog('► Upgrading pip…\n');
  await runStep(pip, ['install', '--upgrade', 'pip']);

  // Step 3 — detect CUDA and install torch
  const cuda   = detectCuda();
  const idxUrl = torchIndexUrl(cuda);
  if (cuda && idxUrl) {
    sendLog(`► CUDA ${cuda[0]}.${cuda[1]} detected — installing torch (GPU)…\n`);
  } else if (cuda) {
    sendLog(`► CUDA ${cuda[0]}.${cuda[1]} detected but too old for GPU torch — installing CPU build…\n`);
  } else {
    sendLog('► No GPU detected — installing torch (CPU)…\n');
  }
  const torchArgs = idxUrl
    ? ['install', 'torch', '--index-url', idxUrl]
    : ['install', 'torch'];
  let torchCode = await runStep(pip, torchArgs);
  if (torchCode !== 0 && idxUrl) {
    // GPU wheel failed (e.g. incompatible GLIBC) — retry with plain CPU build
    sendLog('WARNING: GPU torch install failed — retrying with CPU-only build…\n');
    torchCode = await runStep(pip, ['install', 'torch']);
  }
  if (torchCode !== 0) {
    sendLog('ERROR: torch installation failed (see log above).\n');
    sendDone({ success: false, error: 'torch install failed' });
    return;
  }

  // Step 4 — install ML packages
  sendLog('► Installing ML packages…\n');
  const mlCode = await runStep(pip, ['install', ...ML_PACKAGES]);
  if (mlCode !== 0) {
    sendLog('ERROR: ML packages installation failed (see log above).\n');
    sendDone({ success: false, error: 'ML packages install failed' });
    return;
  }

  // Step 4b — install PanoptoDownloader without deps (avoids yarl~=1.7.2 C++ build on Windows)
  sendLog('► Installing PanoptoDownloader (--no-deps)…\n');
  const panoptoCode = await runStep(pip, ['install', '--no-deps', PANOPTO_PKG]);
  if (panoptoCode !== 0) {
    sendLog('ERROR: PanoptoDownloader installation failed (see log above).\n');
    sendDone({ success: false, error: 'PanoptoDownloader install failed' });
    return;
  }

  // Step 5 — playwright browsers
  sendLog('► Installing Playwright browsers…\n');
  const playwrightCode = await runStep(VENV_PYTHON, ['-m', 'playwright', 'install', 'chromium']);
  if (playwrightCode !== 0) {
    sendLog('WARNING: Playwright browser install failed — video download may not work.\n');
    sendLog('  You can retry from Settings → ML Environment.\n');
    // Non-fatal: other features still work without playwright
  }

  sendLog('\n✓ ML environment ready. You can now run the pipeline.\n');
  sendDone({ success: true });
}

// ── Lecture discovery (for Generate page dropdown) ───────────────────────────
const SLIDE_EXTS = new Set(['.pptx', '.pdf', '.docx', '.ppt']);

function extractLecNum(stem) {
  // L01, L1, L_01, L-01 — must be at start or after a non-letter
  let m = stem.match(/(?:^|[^a-zA-Z])L[_-]?0*(\d+)/i);
  if (m) return parseInt(m[1]);
  // "Lecture 1", "Lecture_2", "Lec 3", "lec_02"
  m = stem.match(/Lec(?:ture)?[_\s-]?0*(\d+)/i);
  if (m) return parseInt(m[1]);
  // leading digits: "05_Scheduling"
  m = stem.match(/^0*(\d+)/);
  if (m) return parseInt(m[1]);
  return null;
}

function walkDir(dir, exts, results = []) {
  if (!fs.existsSync(dir)) return results;
  for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, e.name);
    if (e.isDirectory()) walkDir(full, exts, results);
    else if (exts.has(path.extname(e.name).toLowerCase())) results.push(full);
  }
  return results;
}

function discoverLectures(courseId) {
  const courseDir    = path.join(getOutputDir(), String(courseId));
  const materialsDir = path.join(courseDir, 'materials');
  const alignDir     = path.join(courseDir, 'alignment');

  // Count alignment files so we can show a hint in the dropdown
  let alignCount = 0;
  if (fs.existsSync(alignDir)) {
    alignCount = fs.readdirSync(alignDir)
      .filter(f => f.endsWith('.json') && !f.endsWith('.compact.json') && !f.endsWith('.image_cache.json'))
      .length;
  }

  // Collect slide files recursively, deduplicate by lecture number
  const allFiles = walkDir(materialsDir, SLIDE_EXTS);
  const byNum = new Map();   // num → {num, title, file}
  const noNum = [];

  for (const file of allFiles) {
    const stem  = path.basename(file, path.extname(file));
    const num   = extractLecNum(stem);
    const title = stem.replace(/[-_]/g, ' ').replace(/\s+/g, ' ').trim();
    if (num != null) {
      // Keep the file from the most prominent subdirectory (first found wins)
      if (!byNum.has(num)) byNum.set(num, { num, title, file });
    } else {
      noNum.push({ num: null, title, file });
    }
  }

  const slides = [
    ...[...byNum.values()].sort((a, b) => a.num - b.num),
    ...noNum.sort((a, b) => a.title.localeCompare(b.title)),
  ];

  // Attach whether alignment files exist (Python does the exact matching)
  return slides.map(sl => ({ ...sl, alignCount }));
}

// ── Uninstall helpers ─────────────────────────────────────────────────────────
function getDirSizeMB(dirPath, excludeDirs = []) {
  let bytes = 0;
  try {
    const entries = fs.readdirSync(dirPath, { withFileTypes: true });
    for (const e of entries) {
      if (excludeDirs.includes(e.name)) continue;
      const full = path.join(dirPath, e.name);
      if (e.isDirectory()) {
        bytes += getDirSizeBytes(full);
      } else {
        try { bytes += fs.statSync(full).size; } catch {}
      }
    }
  } catch {}
  return Math.round(bytes / (1024 * 1024));
}

function getDirSizeBytes(dirPath) {
  let bytes = 0;
  try {
    const entries = fs.readdirSync(dirPath, { withFileTypes: true });
    for (const e of entries) {
      const full = path.join(dirPath, e.name);
      if (e.isDirectory()) bytes += getDirSizeBytes(full);
      else try { bytes += fs.statSync(full).size; } catch {}
    }
  } catch {}
  return bytes;
}

function rmRecursive(dirPath) {
  fs.rmSync(dirPath, { recursive: true, force: true });
}

// ── IPC handlers ──────────────────────────────────────────────────────────────
function registerIpc() {
  ipcMain.handle('config:get',      ()       => loadConfig());
  ipcMain.handle('config:set',      (_, d)   => { saveConfig(d); return true; });

  ipcMain.handle('credentials:get', ()       => loadCredentials());
  ipcMain.handle('credentials:set', (_, d)   => { saveCredentials(d); return true; });

  ipcMain.handle('courses:fetch',   ()       => fetchCoursesFromCanvas());

  ipcMain.handle('files:status',    (_, id)  => getFilesStatus(id));

  ipcMain.handle('venv:status',     ()       => ({
    installed: fs.existsSync(VENV_PYTHON),
    path: VENV_PYTHON,
  }));

  ipcMain.handle('python:path',     ()       => getPythonPath());

  ipcMain.handle('constants:get',   (_, { key, name })         => readConstant(key, name));
  ipcMain.handle('constants:set',   (_, { key, name, value })  => writeConstant(key, name, value));

  ipcMain.handle('process:run',     (_, { cmd }) => runProcess(cmd));
  ipcMain.on(    'process:stop',    ()           => stopProcess());

  ipcMain.handle('install:start', async (_, { basePython }) => {
    if (installProc) return { error: 'Installer already running.' };
    runInstaller(
      basePython || null,
      (text)   => mainWindow?.webContents.send('install:data', text),
      (result) => mainWindow?.webContents.send('install:done', result),
    );
    return { ok: true };
  });
  ipcMain.on('install:stop', () => {
    if (installProc) {
      try { installProc.kill('SIGTERM'); } catch {}
      installProc = null;
    }
  });

  ipcMain.handle('scripts:paths',       ()       => SCRIPTS);
  ipcMain.handle('path:outputDir',      ()       => getOutputDir());
  ipcMain.handle('path:dataDir',        ()       => DATA_DIR);
  ipcMain.handle('course:listLectures', (_, cid) => discoverLectures(cid));

  // ── Uninstaller ──────────────────────────────────────────────────────────
  ipcMain.handle('uninstall:sizes', async () => {
    const venvDir   = path.join(DATA_DIR, 'venv');
    const outputDir = getOutputDir();
    return {
      venv:     getDirSizeMB(venvDir),
      settings: getDirSizeMB(DATA_DIR, ['venv']),  // config/scripts/manifest/api-keys, excl venv
      content:  outputDir !== DATA_DIR ? getDirSizeMB(outputDir) : null,
      outputDir,
    };
  });

  ipcMain.handle('uninstall:run', async (_, { keepContent, keepSettings, keepVenv }) => {
    try {
      // 1. Optionally delete ML venv
      if (!keepVenv) {
        const venvDir = path.join(DATA_DIR, 'venv');
        if (fs.existsSync(venvDir)) rmRecursive(venvDir);
      }

      // 2. Optionally delete generated content (OUTPUT_DIR if separate from DATA_DIR)
      if (!keepContent) {
        const outputDir = getOutputDir();
        if (outputDir && outputDir !== DATA_DIR && fs.existsSync(outputDir)) {
          rmRecursive(outputDir);
        }
      }

      // 3. Optionally delete settings (config, API keys, credentials, manifest, scripts cache)
      if (!keepSettings && fs.existsSync(DATA_DIR)) {
        rmRecursive(DATA_DIR);
      } else if (keepSettings) {
        // Still remove the scripts cache dir — it will be re-created if app is reinstalled
        const scriptsDir = path.join(DATA_DIR, 'scripts');
        if (fs.existsSync(scriptsDir)) rmRecursive(scriptsDir);
      }

      // 4. On Windows: launch the NSIS uninstaller so the app is removed from
      //    Windows Settings → Apps automatically, then quit.
      if (process.platform === 'win32') {
        const uninstExe = path.join(path.dirname(process.execPath), 'Uninstall AutoNote.exe');
        if (fs.existsSync(uninstExe)) {
          spawn(uninstExe, ['--updated'], { detached: true, stdio: 'ignore' }).unref();
        }
      }

      return { ok: true, platform: process.platform };
    } catch (err) {
      return { ok: false, error: err.message };
    }
  });

  ipcMain.handle('dialog:openDir', async () => {
    const result = await dialog.showOpenDialog(mainWindow, {
      properties: ['openDirectory'],
    });
    return result.canceled ? null : result.filePaths[0];
  });
}

// ── Window creation ───────────────────────────────────────────────────────────
function createWindow() {
  mainWindow = new BrowserWindow({
    width:     1080,
    height:    780,
    minWidth:  720,
    minHeight: 520,
    show:            false,   // wait for ready-to-show to avoid blank flash
    backgroundColor: '#1E2A2A',
    webPreferences: {
      preload:          path.join(__dirname, 'preload.js'),
      nodeIntegration:  false,
      contextIsolation: true,
    },
  });
  mainWindow.once('ready-to-show', () => mainWindow.show());
  mainWindow.loadFile(path.join(__dirname, 'renderer', 'index.html'));
  if (process.env.AUTONOTE_DEV) mainWindow.webContents.openDevTools();
  // F12 / Ctrl+Shift+I opens DevTools for in-field debugging
  mainWindow.webContents.on('before-input-event', (_, input) => {
    if (input.type !== 'keyDown') return;
    if (input.key === 'F12' ||
        (input.control && input.shift && input.key === 'I')) {
      mainWindow.webContents.openDevTools();
    }
  });
  mainWindow.on('closed', () => { mainWindow = null; });
}

// Disable GPU hardware acceleration on Windows to prevent blank white/teal
// screen that can occur with certain GPU drivers (very common Electron issue).
if (process.platform === 'win32') app.disableHardwareAcceleration();

app.whenReady().then(() => {
  registerIpc();
  createWindow();
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) createWindow();
});

// ── Exports for testing ───────────────────────────────────────────────────────
if (typeof module !== 'undefined') {
  module.exports = {
    loadConfig, saveConfig, getOutputDir, getPythonPath,
    loadCredentials, saveCredentials,
    readConstant, writeConstant,
    getFilesStatus, detectCuda, torchIndexUrl,
    SCRIPTS, DATA_DIR, VENV_PYTHON,
  };
}
