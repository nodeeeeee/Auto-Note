'use strict';

const { app, BrowserWindow, ipcMain, dialog } = require('electron');
const path   = require('path');
const os     = require('os');
const fs     = require('fs');
const https  = require('https');
const http   = require('http');
const { spawnSync, spawn, exec } = require('child_process');

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

function syncScripts() {
  // On first launch and after updates, copy bundled scripts from read-only
  // resources into DATA_DIR/scripts/ so writeConstant() can patch them.
  if (!process.resourcesPath) return;   // dev mode — scripts are already writable
  const srcDir = path.join(process.resourcesPath, 'scripts');
  if (!fs.existsSync(srcDir)) return;
  try {
    if (!fs.existsSync(SCRIPTS_DIR_INSTALLED))
      fs.mkdirSync(SCRIPTS_DIR_INSTALLED, { recursive: true });
    for (const name of fs.readdirSync(srcDir)) {
      const src = path.join(srcDir, name);
      const dst = path.join(SCRIPTS_DIR_INSTALLED, name);
      const srcTxt = fs.readFileSync(src, 'utf8');
      // Overwrite only if missing or if the bundled copy changed (app update).
      if (!fs.existsSync(dst) || fs.readFileSync(dst, 'utf8') !== srcTxt)
        fs.writeFileSync(dst, srcTxt);
    }
  } catch { /* non-fatal */ }
}
syncScripts();  // run before SCRIPTS is built so scriptPath() finds the copies

function scriptPath(name) {
  // 1. Writable copy in DATA_DIR/scripts/ (preferred — supports writeConstant)
  const installed = path.join(SCRIPTS_DIR_INSTALLED, name);
  if (fs.existsSync(installed)) return installed;
  // 2. Packaged read-only resources (fallback before sync completes)
  if (process.resourcesPath) {
    const packed = path.join(process.resourcesPath, 'scripts', name);
    if (fs.existsSync(packed)) return packed;
  }
  // 3. Dev mode: sibling of electron/
  return path.join(PROJECT_DIR, name);
}

const SCRIPTS = {
  downloader:      scriptPath('downloader.py'),
  transcribe:      scriptPath('extract_caption.py'),
  frame_extractor: scriptPath('frame_extractor.py'),
  align:           scriptPath('semantic_alignment.py'),
  generate:        scriptPath('note_generation.py'),
};

// ML packages split into components the user can opt in/out of.
// "core" is always installed; the rest are optional (default: all on).
const ML_COMPONENTS = {
  core: {
    label: 'Core (required)',
    packages: ['tqdm', 'requests', 'pillow', 'httpx', 'openai', 'anthropic',
               'google-generativeai', 'canvasapi', 'pymupdf', 'python-pptx',
               'python-docx', 'ffmpeg-progress-yield', 'pycryptodomex', 'yarl'],
    required: true,
  },
  whisper: {
    label: 'Local transcription (faster-whisper + GPU)',
    packages: ['faster-whisper'],
    required: false,
  },
  embeddings: {
    label: 'Local embeddings (sentence-transformers + FAISS)',
    packages: ['sentence-transformers', 'faiss-cpu'],
    required: false,
  },
  bge_m3: {
    label: 'BGE-M3 model (download ~2 GB on first use)',
    packages: [],   // model is downloaded by sentence-transformers at runtime
    required: false,
    note: 'Requires "Local embeddings" above',
  },
  panopto: {
    label: 'Panopto video download (Playwright browser)',
    packages: [],   // playwright is installed; browser downloaded separately
    required: false,
  },
};

// Flatten for backward compat — used when no component selection is given
const ML_PACKAGES = Object.values(ML_COMPONENTS)
  .flatMap(c => c.packages);
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
  // Normalise PANOPTO_HOST: strip any accidental https:// prefix or trailing path
  if (cfg.PANOPTO_HOST) {
    cfg.PANOPTO_HOST = cfg.PANOPTO_HOST
      .replace(/^https?:\/\//i, '')
      .split('/')[0]
      .trim();
  }
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
    deepseek:  read('deepseek_key.txt'),
    grok:      read('grok_key.txt'),
    mistral:   read('mistral_key.txt'),
  };
}

function saveCredentials(data) {
  ensureDataDir();
  const map = {
    canvas:    'canvas_token.txt',
    openai:    'openai_api.txt',
    anthropic: 'anthropic_key.txt',
    gemini:    'gemini_api.txt',
    deepseek:  'deepseek_key.txt',
    grok:      'grok_key.txt',
    mistral:   'mistral_key.txt',
  };
  for (const [key, file] of Object.entries(map)) {
    // Strip ALL whitespace (including internal newlines) — guards against
    // accidental paste of multiple tokens or copy with trailing newlines.
    const val = (data[key] || '').replace(/\s+/g, '');
    if (val) fs.writeFileSync(path.join(DATA_DIR, file), val);
  }
}

// ── Canvas API ────────────────────────────────────────────────────────────────
function httpGetWithHeaders(url, headers) {
  return new Promise((resolve, reject) => {
    const mod = url.startsWith('https') ? https : http;
    const req = mod.get(url, { headers }, (res) => {
      let body = '';
      res.on('data', d => body += d);
      res.on('end', () => {
        if (res.statusCode === 401) {
          return reject(new Error('401 Unauthorized — Canvas token is invalid or expired.'));
        }
        if (res.statusCode >= 400) {
          return reject(new Error(`HTTP ${res.statusCode}: ${url}`));
        }
        try {
          resolve({ data: JSON.parse(body), linkHeader: res.headers['link'] || '' });
        } catch {
          reject(new Error('Invalid JSON response from server'));
        }
      });
    });
    req.on('error', reject);
    req.setTimeout(12000, () => { req.destroy(); reject(new Error('Request timed out')); });
  });
}

function httpGet(url, headers) {
  return httpGetWithHeaders(url, headers).then(r => r.data);
}

// Parse the Canvas `Link` response header to find the next-page URL.
function parseLinkNext(linkHeader) {
  if (!linkHeader) return null;
  for (const part of linkHeader.split(',')) {
    const m = part.match(/<([^>]+)>.*rel="next"/);
    if (m) return m[1];
  }
  return null;
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
  let baseUrl = (cfg.CANVAS_URL || '').trim().replace(/\/$/, '');
  if (!baseUrl) return { error: 'Canvas URL not configured — enter it in Settings → Connection.' };
  if (!baseUrl.startsWith('http')) baseUrl = 'https://' + baseUrl;
  try {
    // Fetch all courses the user has any enrollment in (student, TA, auditor, cross-listed…).
    // Do NOT filter by enrollment_state=active — that drops TAs, design-experience tracks,
    // and cross-listed courses whose enrollment state differs from "active".
    // Follow Link headers for pagination in case the user has >100 courses.
    const allData = [];
    const authHeaders = { 'Authorization': `Bearer ${token}` };
    let nextUrl = `${baseUrl}/api/v1/courses?per_page=100`;
    let pages = 0;
    while (nextUrl && pages < 20) {   // safety cap: 20 pages = 2000 courses max
      const { data, linkHeader } = await httpGetWithHeaders(nextUrl, authHeaders);
      allData.push(...(Array.isArray(data) ? data : []));
      nextUrl = parseLinkNext(linkHeader);
      pages++;
    }
    const courses = allData
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
    if (!re.test(src)) return false;           // constant not found in script
    const newSrc = src.replace(re, `$1${newVal}`);
    if (newSrc !== src) fs.writeFileSync(sp, newSrc);  // skip write if unchanged
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
async function runInstaller(basePython, sendLog, sendDone, selectedComponents = null) {
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

  // Determine which components to install
  const sel = selectedComponents || Object.keys(ML_COMPONENTS);
  const needsTorch = sel.includes('whisper') || sel.includes('embeddings');
  sendLog(`► Components: ${sel.join(', ')}\n`);

  // Step 3 — detect CUDA and install torch (only if needed)
  if (needsTorch) {
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
      sendLog('WARNING: GPU torch install failed — retrying with CPU-only build…\n');
      torchCode = await runStep(pip, ['install', 'torch']);
    }
    if (torchCode !== 0) {
      sendLog('ERROR: torch installation failed (see log above).\n');
      sendDone({ success: false, error: 'torch install failed' });
      return;
    }
  } else {
    sendLog('► Skipping torch (no local ML components selected).\n');
  }

  // Step 4 — install selected packages
  const packages = [];
  for (const key of sel) {
    const comp = ML_COMPONENTS[key];
    if (comp && comp.packages.length) packages.push(...comp.packages);
  }
  // Always include core
  if (!sel.includes('core')) packages.push(...ML_COMPONENTS.core.packages);

  sendLog(`► Installing ${packages.length} package(s)…\n`);
  const mlCode = await runStep(pip, ['install', ...packages]);
  if (mlCode !== 0) {
    sendLog('ERROR: ML packages installation failed (see log above).\n');
    sendDone({ success: false, error: 'ML packages install failed' });
    return;
  }

  // Step 4b — install PanoptoDownloader (only if panopto component selected)
  if (sel.includes('panopto')) {
    sendLog('► Installing Playwright + PanoptoDownloader…\n');
    await runStep(pip, ['install', 'playwright']);
    const panoptoCode = await runStep(pip, ['install', '--no-deps', PANOPTO_PKG]);
    if (panoptoCode !== 0) {
      sendLog('WARNING: PanoptoDownloader install failed — video download may not work.\n');
    }
    // Step 5 — playwright browsers
    sendLog('► Installing Playwright browsers…\n');
    const playwrightCode = await runStep(VENV_PYTHON, ['-m', 'playwright', 'install', 'chromium']);
    if (playwrightCode !== 0) {
      sendLog('WARNING: Playwright browser install failed — video download may not work.\n');
      sendLog('  You can retry from Settings → ML Environment.\n');
    }
  } else {
    sendLog('► Skipping Panopto/Playwright (not selected).\n');
  }

  // Step 5b — pre-download BGE-M3 model if selected
  if (sel.includes('bge_m3') && sel.includes('embeddings')) {
    sendLog('► Pre-downloading BGE-M3 model (~2 GB)…\n');
    const dlCode = await runStep(VENV_PYTHON, ['-c',
      "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"]);
    if (dlCode !== 0) {
      sendLog('WARNING: BGE-M3 download failed — it will download on first use.\n');
    }
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

// ── Course video listing & deletion ──────────────────────────────────────────

function listCourseVideos(courseId) {
  const courseDir  = path.join(getOutputDir(), String(courseId));
  const capDir     = path.join(courseDir, 'captions');
  const alignDir   = path.join(courseDir, 'alignment');
  const notesDir   = path.join(courseDir, 'notes');
  const sectionsDir = path.join(notesDir, 'sections');

  if (!fs.existsSync(capDir)) return [];

  const captions = fs.readdirSync(capDir)
    .filter(f => f.endsWith('.json'))
    .sort();

  // Build map: captionStem → [{file, slideFile, lecNum}]
  const alignMap = {};
  if (fs.existsSync(alignDir)) {
    for (const af of fs.readdirSync(alignDir)) {
      if (!af.endsWith('.json') || af.endsWith('.compact.json')
          || af.includes('mapping') || af.includes('image_cache')) continue;
      try {
        const data = JSON.parse(fs.readFileSync(path.join(alignDir, af), 'utf8'));
        const lecStem = data.lecture || '';
        if (!alignMap[lecStem]) alignMap[lecStem] = [];
        alignMap[lecStem].push({
          file: af,
          slideFile: data.slide_file || '',
          lecNum: extractLecNum(data.slide_file || af),
        });
      } catch { /* skip unreadable files */ }
    }
  }

  // List note section files
  const sectionFiles = fs.existsSync(sectionsDir)
    ? fs.readdirSync(sectionsDir).filter(f => f.endsWith('.md'))
    : [];

  // List per-video note files
  const noteFiles = fs.existsSync(notesDir)
    ? fs.readdirSync(notesDir).filter(f => f.endsWith('.md'))
    : [];

  return captions.map(f => {
    const stem = f.replace(/\.json$/, '');
    const aligns = alignMap[stem] || [];
    const hasAlignment = aligns.length > 0;

    // Check for notes: section files matching any associated lecture number
    const lecNums = aligns.map(a => a.lecNum).filter(n => n != null);
    const hasNotes = lecNums.some(num => {
      const prefix = `L${String(num).padStart(2, '0')}_S`;
      return sectionFiles.some(sf => sf.startsWith(prefix));
    });

    return { stem, filename: f, hasAlignment, hasNotes, lecNums };
  });
}

function deleteVideoData(courseId, captionStem) {
  const courseDir  = path.join(getOutputDir(), String(courseId));
  const capDir     = path.join(courseDir, 'captions');
  const alignDir   = path.join(courseDir, 'alignment');
  const notesDir   = path.join(courseDir, 'notes');
  const sectionsDir = path.join(notesDir, 'sections');
  const imagesDir  = path.join(notesDir, 'images');

  const deleted = [];

  // 1. Delete caption file
  const capFile = path.join(capDir, captionStem + '.json');
  if (fs.existsSync(capFile)) {
    fs.unlinkSync(capFile);
    deleted.push('captions/' + captionStem + '.json');
  }

  // 2. Find and delete alignment files (by name match or lecture field)
  const lecNums = new Set();
  if (fs.existsSync(alignDir)) {
    for (const af of fs.readdirSync(alignDir)) {
      if (!af.endsWith('.json') || af.includes('mapping') || af.includes('image_cache')) continue;
      const afPath = path.join(alignDir, af);

      // Direct name match (caption stem)
      if (af === captionStem + '.json' || af === captionStem + '.compact.json') {
        // Read lecture number before deleting
        if (!af.endsWith('.compact.json')) {
          try {
            const data = JSON.parse(fs.readFileSync(afPath, 'utf8'));
            const num = extractLecNum(data.slide_file || af);
            if (num != null) lecNums.add(num);
          } catch {}
        }
        fs.unlinkSync(afPath);
        deleted.push('alignment/' + af);
        continue;
      }

      // Check lecture field for multi-slide outputs (named after slide stem)
      if (!af.endsWith('.compact.json')) {
        try {
          const data = JSON.parse(fs.readFileSync(afPath, 'utf8'));
          if (data.lecture === captionStem) {
            const num = extractLecNum(data.slide_file || af);
            if (num != null) lecNums.add(num);
            fs.unlinkSync(afPath);
            deleted.push('alignment/' + af);
            // Also delete compact version
            const compactPath = afPath.replace('.json', '.compact.json');
            if (fs.existsSync(compactPath)) {
              fs.unlinkSync(compactPath);
              deleted.push('alignment/' + af.replace('.json', '.compact.json'));
            }
          }
        } catch {}
      }
    }
  }

  // 3. Delete related note sections and per-video notes
  for (const num of lecNums) {
    const prefix = `L${String(num).padStart(2, '0')}`;

    // Delete section files: L{num}_S*.md and L{num}_F*_S*.md
    if (fs.existsSync(sectionsDir)) {
      for (const sf of fs.readdirSync(sectionsDir)) {
        if (sf.startsWith(prefix + '_') && sf.endsWith('.md')) {
          fs.unlinkSync(path.join(sectionsDir, sf));
          deleted.push('notes/sections/' + sf);
        }
      }
    }

    // Delete per-video note files: *_L{num}_notes.*
    if (fs.existsSync(notesDir)) {
      for (const nf of fs.readdirSync(notesDir)) {
        if (nf.includes(`_${prefix}_notes`) && !fs.statSync(path.join(notesDir, nf)).isDirectory()) {
          fs.unlinkSync(path.join(notesDir, nf));
          deleted.push('notes/' + nf);
        }
      }
    }

    // Delete images directories: images/L{num}/ and images/L{num}_F*/
    if (fs.existsSync(imagesDir)) {
      for (const d of fs.readdirSync(imagesDir)) {
        if ((d === prefix || d.startsWith(prefix + '_F')) &&
            fs.statSync(path.join(imagesDir, d)).isDirectory()) {
          fs.rmSync(path.join(imagesDir, d), { recursive: true, force: true });
          deleted.push('notes/images/' + d + '/');
        }
      }
    }
  }

  // 4. Delete merged notes file if any lecture was removed (it needs regeneration)
  if (lecNums.size > 0 && fs.existsSync(notesDir)) {
    for (const nf of fs.readdirSync(notesDir)) {
      if (nf.endsWith('_notes.md') && !nf.includes('_L')) {
        fs.unlinkSync(path.join(notesDir, nf));
        deleted.push('notes/' + nf);
        // Also delete score file
        const scoreFile = path.join(notesDir, nf.replace('.md', '.score.json'));
        if (fs.existsSync(scoreFile)) {
          fs.unlinkSync(scoreFile);
          deleted.push('notes/' + nf.replace('.md', '.score.json'));
        }
      }
    }
  }

  return { deleted, lecNums: [...lecNums] };
}

// ── Uninstall helpers ─────────────────────────────────────────────────────────
// Returns directory size in MB using native OS tools (non-blocking, fast).
function getDirSizeMBNative(dirPath) {
  return new Promise(resolve => {
    if (!fs.existsSync(dirPath)) { resolve(0); return; }
    if (process.platform === 'win32') {
      // PowerShell: sum all file lengths under the directory
      const esc = dirPath.replace(/'/g, "''");
      const cmd = `powershell -NoProfile -NonInteractive -Command ` +
        `"(Get-ChildItem -LiteralPath '${esc}' -Recurse -File ` +
        `-ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum"`;
      exec(cmd, { timeout: 30000, windowsHide: true }, (_, stdout) => {
        const bytes = parseInt((stdout || '').trim()) || 0;
        resolve(Math.round(bytes / 1048576));
      });
    } else {
      // Linux / macOS: du -sm gives megabytes directly
      const esc = dirPath.replace(/\\/g, '\\\\').replace(/"/g, '\\"');
      exec(`du -sm -- "${esc}"`, { timeout: 30000 }, (err, stdout) => {
        if (err) { resolve(0); return; }
        const mb = parseInt((stdout || '').trim().split(/\s+/)[0]) || 0;
        resolve(mb);
      });
    }
  });
}

async function getDirSizeMB(dirPath, excludeDirs = []) {
  if (!fs.existsSync(dirPath)) return 0;
  const total = await getDirSizeMBNative(dirPath);
  if (excludeDirs.length === 0) return total;
  // Subtract excluded subdirs from the total
  const excluded = await Promise.all(
    excludeDirs
      .map(d => path.join(dirPath, d))
      .filter(p => fs.existsSync(p))
      .map(getDirSizeMBNative)
  );
  return Math.max(0, total - excluded.reduce((a, b) => a + b, 0));
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

  // Expose component definitions to renderer
  ipcMain.handle('install:components', () => ML_COMPONENTS);

  ipcMain.handle('install:start', async (_, { basePython, components }) => {
    if (installProc) return { error: 'Installer already running.' };
    runInstaller(
      basePython || null,
      (text)   => mainWindow?.webContents.send('install:data', text),
      (result) => mainWindow?.webContents.send('install:done', result),
      components || null,  // null = install all
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
  ipcMain.handle('course:listVideos',   (_, cid) => listCourseVideos(cid));
  ipcMain.handle('course:deleteVideo',  (_, { cid, stem }) => deleteVideoData(cid, stem));

  // ── Align: scan captions + slides, load/save mapping ───────────────────
  ipcMain.handle('align:scan', (_, cid) => {
    // Course data lives under OUTPUT_DIR (~/AutoNote by default).
    // Check OUTPUT_DIR first, then DATA_DIR as fallback.
    const candidates = [
      path.join(getOutputDir(), String(cid)),
      path.join(DATA_DIR, String(cid)),
    ];
    let base = candidates[0];  // default to OUTPUT_DIR
    for (const c of candidates) {
      if (fs.existsSync(path.join(c, 'captions')) || fs.existsSync(path.join(c, 'materials'))) {
        base = c; break;
      }
    }
    const capDir    = path.join(base, 'captions');
    const matDir    = path.join(base, 'materials');
    const alignDir  = path.join(base, 'alignment');
    const exts      = new Set(['.pdf', '.pptx', '.ppt', '.docx', '.doc']);

    // Captions
    let captions = [];
    if (fs.existsSync(capDir)) {
      captions = fs.readdirSync(capDir)
        .filter(f => f.endsWith('.json'))
        .sort()
        .map(f => {
          const stem = f.replace(/\.json$/, '');
          const aligned = fs.existsSync(path.join(alignDir, f));
          return { stem, filename: f, aligned };
        });
    }

    // Slides (recursive)
    const slides = [];
    function walkDir(dir, rel) {
      if (!fs.existsSync(dir)) return;
      for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
        if (entry.isDirectory()) {
          walkDir(path.join(dir, entry.name), rel ? rel + '/' + entry.name : entry.name);
        } else if (exts.has(path.extname(entry.name).toLowerCase()) && !entry.name.includes('image_cache')) {
          const relPath = rel ? 'materials/' + rel + '/' + entry.name : 'materials/' + entry.name;
          slides.push({ name: entry.name, rel: relPath });
        }
      }
    }
    walkDir(matDir, '');

    // Video titles from manifest
    const titles = {};
    const mf = path.join(DATA_DIR, 'manifest.json');
    if (fs.existsSync(mf)) {
      try {
        const manifest = JSON.parse(fs.readFileSync(mf, 'utf8'));
        for (const [, entry] of Object.entries(manifest)) {
          if (entry.status === 'done' && entry.title) {
            const sanitized = entry.title.replace(/[\\/*?:"<>|]/g, '_');
            titles[sanitized] = entry.title;
          }
        }
      } catch {}
    }

    // Existing mapping
    let mapping = {};
    const mapFile = path.join(alignDir, 'video_slide_mapping.json');
    if (fs.existsSync(mapFile)) {
      try { mapping = JSON.parse(fs.readFileSync(mapFile, 'utf8')); } catch {}
    }

    // ── Smart auto-suggest for each caption ──────────────────────────────────
    // Strategy: lecture-number match → date match → transcript keyword overlap
    function autoSuggest(capStem) {
      const cl = capStem.toLowerCase().replace(/[-_]/g, ' ');

      // 1. Extract number from caption: Week3, Lec2, L5, etc.
      const numMatch = cl.match(/week\s*(\d+)|lec(?:ture)?\s*(\d+)|\bl(\d+)\b/);
      const capNum = numMatch ? (numMatch[1] || numMatch[2] || numMatch[3]) : null;

      // 2. Extract date from caption: "06/03/2026" or "06_03_2026"
      const dateMatch = capStem.match(/(\d{2})[/_](\d{2})[/_](\d{4})/);
      const capDate = dateMatch ? dateMatch[1] + dateMatch[2] + dateMatch[3].slice(2) : null;

      // 3. Read first few lines of transcript for keyword extraction
      let capKeywords = new Set();
      const capFile = path.join(capDir, capStem + '.json');
      if (fs.existsSync(capFile)) {
        try {
          const capData = JSON.parse(fs.readFileSync(capFile, 'utf8'));
          const segs = capData.segments || [];
          // Sample ~20 segments evenly for topic keywords
          const step = Math.max(1, Math.floor(segs.length / 20));
          const words = segs.filter((_, i) => i % step === 0)
            .map(s => s.text || '').join(' ').toLowerCase()
            .replace(/[^a-z0-9\s]/g, '').split(/\s+/)
            .filter(w => w.length > 4);
          capKeywords = new Set(words);
        } catch {}
      }

      let bestScore = 0;
      let bestSlide = null;

      for (const s of slides) {
        const sl = s.name.toLowerCase().replace(/[-_]/g, ' ');
        let score = 0;

        // Strategy 1: Lecture number match
        const slNumMatch = sl.match(/l(?:ecture)?\s*(\d+)/);
        if (capNum && slNumMatch && capNum === slNumMatch[1]) {
          score += 100;
        }

        // Strategy 2: Date match (video date → slide annotation date)
        // e.g. "06/03/2026" matches "ann060326" in slide filename
        if (capDate) {
          if (sl.includes(capDate) || s.name.includes(capDate)) {
            score += 80;
          }
        }

        // Strategy 3: Token overlap between caption stem and slide name
        const capTokens = new Set(cl.split(/\s+/).filter(t => t.length > 1));
        const slTokens = new Set(sl.split(/\s+/).filter(t => t.length > 1));
        const inter = [...capTokens].filter(t => slTokens.has(t)).length;
        if (capTokens.size && slTokens.size) {
          score += (inter / Math.max(capTokens.size, slTokens.size)) * 30;
        }

        // Strategy 4: Transcript keyword overlap with slide filename words
        if (capKeywords.size > 0) {
          const slWords = sl.replace(/[^a-z0-9\s]/g, '').split(/\s+/).filter(w => w.length > 4);
          const kwHits = slWords.filter(w => capKeywords.has(w)).length;
          score += kwHits * 10;
        }

        // Preference: "with notes" > "review" > annotated > plain
        if (/with\s*notes/i.test(s.name)) score += 5;
        else if (/review/i.test(s.name)) score += 3;
        else if (/ann\d/i.test(s.name)) score += 2;

        // Preference: files in a "Lectures" subfolder
        if (/lecture/i.test(s.rel)) score += 2;

        if (score > bestScore) {
          bestScore = score;
          bestSlide = s.rel;
        }
      }

      return { suggested: bestScore > 5 ? bestSlide : null, score: bestScore };
    }

    // Attach suggestions to captions
    for (const cap of captions) {
      const { suggested } = autoSuggest(cap.stem);
      cap.suggested = suggested;
    }

    return { captions, slides, titles, mapping, base };
  });

  // Embedding-based smart matching — calls Python semantic_alignment.py --suggest-matches
  ipcMain.handle('align:suggestMatches', async (_, { cid, model }) => {
    const python = getPythonPath();
    const script = SCRIPTS.align;
    const outDir = getOutputDir();
    const courseDir = path.join(outDir, String(cid));

    // Pre-flight diagnostics
    const diagLines = [
      `Python: ${python} (exists: ${fs.existsSync(python)})`,
      `Script: ${script} (exists: ${fs.existsSync(script)})`,
      `DATA_DIR: ${DATA_DIR}`,
      `OutputDir: ${outDir}`,
      `CourseDir: ${courseDir} (exists: ${fs.existsSync(courseDir)})`,
      `Captions dir: ${fs.existsSync(path.join(courseDir, 'captions'))}`,
      `Materials dir: ${fs.existsSync(path.join(courseDir, 'materials'))}`,
    ];

    // Quick pre-check: can this Python import the needed packages?
    const checkCode = "import sys; " +
      "errs=[]; " +
      "[errs.append(n) for n in ['numpy','fitz','sentence_transformers'] if not __import__('importlib').util.find_spec(n)]; " +
      "print('MISSING:'+','.join(errs) if errs else 'OK'); " +
      "print('PYTHON:'+sys.executable); " +
      "print('PREFIX:'+sys.prefix)";
    const chk = spawnSync(python, ['-c', checkCode], {
      encoding: 'utf8', timeout: 10000,
      env: { ...process.env, AUTONOTE_DATA_DIR: DATA_DIR },
    });
    const chkOut = (chk.stdout || '') + (chk.stderr || '');
    diagLines.push(`PreCheck: ${chkOut.replace(/\n/g, ' | ').trim()}`);
    const diagStr = diagLines.join('\n');
    console.log(`[align:suggestMatches]\n${diagStr}`);

    // If packages are missing, return immediately with clear error
    if (chkOut.includes('MISSING:') && !chkOut.includes('OK')) {
      const missing = chkOut.match(/MISSING:(.+)/)?.[1]?.trim();
      if (missing) {
        return { __error: `Python (${python}) is missing packages: ${missing}.\nReinstall ML environment with these components enabled.\n\n${diagStr}` };
      }
    }

    // Run the actual matching script (synchronous — takes a few seconds)
    const args = [script, '--course', String(cid),
                  '--suggest-matches', '--match-model', model || 'bge-m3'];

    // Check the script file actually has --suggest-matches (old versions don't)
    try {
      const scriptContent = fs.readFileSync(script, 'utf8');
      if (!scriptContent.includes('suggest-matches')) {
        diagLines.push(`Script OUTDATED: ${script} does not contain --suggest-matches`);
        // Force-copy the latest script from bundled resources
        if (process.resourcesPath) {
          const bundled = path.join(process.resourcesPath, 'scripts', 'semantic_alignment.py');
          if (fs.existsSync(bundled)) {
            fs.copyFileSync(bundled, script);
            diagLines.push('Copied latest script from AppImage resources.');
          }
        }
      }
    } catch (e) {
      diagLines.push(`Cannot read script: ${e.message}`);
    }

    diagLines.push(`Command: ${python} ${args.join(' ')}`);
    const diagStr2 = diagLines.join('\n');

    let result;
    try {
      result = spawnSync(python, args, {
        encoding: 'utf8',
        timeout: 120000,  // 2 minutes max
        env: { ...process.env, AUTONOTE_DATA_DIR: DATA_DIR },
      });
    } catch (e) {
      return { __error: `Failed to run: ${e.message}\n\n── Diagnostics ──\n${diagStr2}` };
    }

    // Full diagnostics from spawnSync
    const stdout = result.stdout || '';
    const stderr = result.stderr || '';
    const combined = stdout + '\n' + stderr;
    const spawnInfo = `status=${result.status}, signal=${result.signal}, error=${result.error || 'none'}`;
    diagLines.push(`SpawnSync: ${spawnInfo}`);
    diagLines.push(`Stdout length: ${stdout.length}, Stderr length: ${stderr.length}`);

    if (result.error) {
      return { __error: `Spawn error: ${result.error.message}\n\n── Diagnostics ──\n${diagLines.join('\n')}` };
    }

    if (result.status !== 0) {
      return { __error: `Exit code ${result.status}:\n${combined.trim()}\n\n── Diagnostics ──\n${diagLines.join('\n')}` };
    }

    const marker = '__MATCH_RESULT__';
    const idx = combined.indexOf(marker);
    if (idx < 0) {
      return { __error: `No results marker in output.\n\nFull output:\n${combined.trim()}\n\n── Diagnostics ──\n${diagLines.join('\n')}` };
    }

    try {
      const parsed = JSON.parse(combined.slice(idx + marker.length).trim());
      parsed.__log = combined.slice(0, idx).trim();
      return parsed;
    } catch (e) {
      return { __error: `JSON parse error: ${e.message}\n${combined.trim()}` };
    }
  });

  ipcMain.handle('align:saveMapping', (_, { cid, mapping }) => {
    const candidates = [
      path.join(getOutputDir(), String(cid)),
      path.join(DATA_DIR, String(cid)),
    ];
    let base = candidates[0];
    for (const c of candidates) {
      if (fs.existsSync(path.join(c, 'captions')) || fs.existsSync(path.join(c, 'materials'))) {
        base = c; break;
      }
    }
    const alignDir = path.join(base, 'alignment');
    if (!fs.existsSync(alignDir)) fs.mkdirSync(alignDir, { recursive: true });
    const mapFile  = path.join(alignDir, 'video_slide_mapping.json');
    fs.writeFileSync(mapFile, JSON.stringify(mapping, null, 2));
    return mapFile;
  });

  // ── Uninstaller ──────────────────────────────────────────────────────────
  ipcMain.handle('uninstall:sizes', async () => {
    const venvDir   = path.join(DATA_DIR, 'venv');
    const outputDir = getOutputDir();
    const [venv, settings, content] = await Promise.all([
      getDirSizeMB(venvDir),
      getDirSizeMB(DATA_DIR, ['venv']),
      outputDir !== DATA_DIR ? getDirSizeMB(outputDir) : Promise.resolve(null),
    ]);
    return { venv, settings, content, outputDir };
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
