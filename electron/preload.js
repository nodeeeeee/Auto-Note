'use strict';

const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('api', {
  // ── Config ────────────────────────────────────────────────────────────────
  getConfig:      ()     => ipcRenderer.invoke('config:get'),
  setConfig:      (data) => ipcRenderer.invoke('config:set', data),

  // ── Credentials ───────────────────────────────────────────────────────────
  getCredentials: ()     => ipcRenderer.invoke('credentials:get'),
  setCredentials: (data) => ipcRenderer.invoke('credentials:set', data),

  // ── Courses (Canvas API) ──────────────────────────────────────────────────
  fetchCourses:   ()     => ipcRenderer.invoke('courses:fetch'),

  // ── File status (Dashboard) ───────────────────────────────────────────────
  getFilesStatus: (courseId) => ipcRenderer.invoke('files:status', courseId),

  // ── Venv / Python ─────────────────────────────────────────────────────────
  getVenvStatus:  ()     => ipcRenderer.invoke('venv:status'),
  getPythonPath:  ()     => ipcRenderer.invoke('python:path'),

  // ── Script constants (Settings) ───────────────────────────────────────────
  getConstant:    (key, name)        => ipcRenderer.invoke('constants:get', { key, name }),
  setConstant:    (key, name, value) => ipcRenderer.invoke('constants:set', { key, name, value }),

  // ── Subprocess management ─────────────────────────────────────────────────
  runProcess:     (cmd)  => ipcRenderer.invoke('process:run', { cmd }),
  stopProcess:    ()     => ipcRenderer.send('process:stop'),
  onProcessData:  (cb)   => ipcRenderer.on('process:data', (_, text) => cb(text)),
  onProcessDone:  (cb)   => ipcRenderer.on('process:done', (_, info) => cb(info)),
  offProcessEvents: ()   => {
    ipcRenderer.removeAllListeners('process:data');
    ipcRenderer.removeAllListeners('process:done');
  },

  // ── ML environment installer ──────────────────────────────────────────────
  startInstall:   (basePython) => ipcRenderer.invoke('install:start', { basePython }),
  stopInstall:    ()     => ipcRenderer.send('install:stop'),
  onInstallData:  (cb)   => ipcRenderer.on('install:data', (_, text) => cb(text)),
  onInstallDone:  (cb)   => ipcRenderer.on('install:done', (_, info) => cb(info)),
  offInstallEvents: ()   => {
    ipcRenderer.removeAllListeners('install:data');
    ipcRenderer.removeAllListeners('install:done');
  },

  // ── Script paths ──────────────────────────────────────────────────────────
  getScriptsPaths: ()    => ipcRenderer.invoke('scripts:paths'),

  // ── Course helpers ────────────────────────────────────────────────────────
  listLectures:   (cid)  => ipcRenderer.invoke('course:listLectures', cid),

  // ── Align: scan + mapping ───────────────────────────────────────────────
  alignScan:           (cid)               => ipcRenderer.invoke('align:scan', cid),
  alignSuggestMatches: (cid, model)       => ipcRenderer.invoke('align:suggestMatches', { cid, model }),
  alignSaveMapping:    (cid, mapping)     => ipcRenderer.invoke('align:saveMapping', { cid, mapping }),

  // ── OS/Dialog helpers ─────────────────────────────────────────────────────
  openDirDialog:  ()     => ipcRenderer.invoke('dialog:openDir'),
  getOutputDir:   ()     => ipcRenderer.invoke('path:outputDir'),
  getDataDir:     ()     => ipcRenderer.invoke('path:dataDir'),

  // ── Uninstaller ───────────────────────────────────────────────────────────
  getUninstallSizes: ()                             => ipcRenderer.invoke('uninstall:sizes'),
  runUninstall:      (keepContent, keepSettings, keepVenv) => ipcRenderer.invoke('uninstall:run', { keepContent, keepSettings, keepVenv }),
});
