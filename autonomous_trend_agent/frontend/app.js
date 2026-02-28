/**
 * AI Video Pipeline — Dashboard App
 * Real-time WebSocket client with progress tracking, system monitoring,
 * pipeline control, and output gallery.
 * 
 * All data is REAL-TIME from the server. No fake/placeholder data.
 */

(function () {
    'use strict';

    // ========================================================================
    // Configuration
    // ========================================================================
    const WS_URL = `ws://${location.host}/ws`;
    const API = `/api`;
    const SYSTEM_POLL_MS = 2000;  // Poll system stats every 2s
    const RECONNECT_MS = 3000;

    // ========================================================================
    // State
    // ========================================================================
    let ws = null;
    let wsConnected = false;
    let systemPollTimer = null;
    let pipelineStartTime = null;
    let elapsedTimer = null;

    // ========================================================================
    // DOM References
    // ========================================================================
    const $ = (sel) => document.querySelector(sel);
    const $$ = (sel) => document.querySelectorAll(sel);

    const dom = {
        // Header badges
        gpuBadge: $('#gpu-badge'),
        gpuBadgeText: $('#gpu-badge .badge-text'),
        wsBadge: $('#ws-badge'),
        wsBadgeText: $('#ws-badge .badge-text'),

        // Control panel
        pipelineState: $('#pipeline-state'),
        videoSelect: $('#video-select'),
        customPath: $('#custom-path'),
        numShorts: $('#num-shorts'),
        toggleCaptions: $('#toggle-captions'),
        toggleEffects: $('#toggle-effects'),
        toggleTracking: $('#toggle-tracking'),
        toggleAnalysis: $('#toggle-analysis'),
        btnStart: $('#btn-start'),
        btnPause: $('#btn-pause'),
        btnStop: $('#btn-stop'),
        btnRefreshVideos: $('#btn-refresh-videos'),

        // Progress
        overallPct: $('#overall-pct'),
        overallBar: $('#overall-bar'),
        elapsedTime: $('#elapsed-time'),
        etaTime: $('#eta-time'),

        // System
        gpuUtilBar: $('#gpu-util-bar'),
        gpuUtilText: $('#gpu-util-text'),
        vramBar: $('#vram-bar'),
        vramText: $('#vram-text'),
        ramBar: $('#ram-bar'),
        ramText: $('#ram-text'),
        gpuTempText: $('#gpu-temp-text'),

        // Log
        logOutput: $('#log-output'),
        btnClearLog: $('#btn-clear-log'),

        // Outputs
        outputsList: $('#outputs-list'),
        btnRefreshOutputs: $('#btn-refresh-outputs'),
    };

    // ========================================================================
    // WebSocket Connection
    // ========================================================================
    function connectWS() {
        if (ws && ws.readyState <= 1) return;

        ws = new WebSocket(WS_URL);

        ws.onopen = () => {
            wsConnected = true;
            setWSBadge(true);
            addLog('Connected to pipeline server', 'system');
        };

        ws.onclose = () => {
            wsConnected = false;
            setWSBadge(false);
            setTimeout(connectWS, RECONNECT_MS);
        };

        ws.onerror = () => {
            wsConnected = false;
            setWSBadge(false);
        };

        ws.onmessage = (evt) => {
            try {
                const data = JSON.parse(evt.data);
                handleWSEvent(data);
            } catch (e) {
                console.error('WS parse error:', e);
            }
        };
    }

    function handleWSEvent(data) {
        switch (data.type) {
            case 'snapshot':
                applySnapshot(data);
                break;
            case 'state_change':
                setPipelineState(data.state);
                break;
            case 'stage_start':
                onStageStart(data.stage, data.total_steps);
                updateOverall(data.overall_pct);
                addLog(`[STAGE] ${data.stage}`, 'stage');
                break;
            case 'step':
                onStep(data.stage, data.step, data.current, data.total, data.stage_pct);
                updateOverall(data.overall_pct);
                if (data.total > 0) {
                    addLog(`  [${data.current}/${data.total}] (${Math.round(data.stage_pct)}%) ${data.step}`, 'step');
                } else {
                    addLog(`  → ${data.step}`, 'step');
                }
                break;
            case 'stage_complete':
                onStageComplete(data.stage);
                updateOverall(data.overall_pct);
                addLog(`  ✅ ${data.stage} complete`, 'success');
                break;
            case 'error':
                onError(data.stage, data.error);
                updateOverall(data.overall_pct);
                addLog(`  ❌ ${data.stage}: ${data.error}`, 'error');
                break;
        }
    }

    function applySnapshot(snap) {
        setPipelineState(snap.state);
        updateOverall(snap.overall_pct);

        // Restore phase states
        for (const stage of (snap.completed_stages || [])) {
            setPhaseStatus(stage, 'complete', 100);
        }
        for (const [stage, prog] of Object.entries(snap.stage_progress || {})) {
            if (snap.completed_stages && snap.completed_stages.includes(stage)) continue;
            setPhaseStatus(stage, 'running', prog.pct || 0, prog.step);
        }

        // Restore log
        if (snap.log_lines) {
            for (const line of snap.log_lines) {
                let cls = 'step';
                if (line.startsWith('[STAGE]')) cls = 'stage';
                else if (line.includes('✅')) cls = 'success';
                else if (line.includes('❌')) cls = 'error';
                addLog(line, cls, false);
            }
        }

        if (snap.elapsed_seconds && snap.state === 'running') {
            pipelineStartTime = Date.now() - snap.elapsed_seconds * 1000;
        }
    }

    // ========================================================================
    // UI Updates
    // ========================================================================
    function setWSBadge(connected) {
        dom.wsBadge.className = `status-badge ${connected ? 'online' : 'offline'}`;
        dom.wsBadgeText.textContent = connected ? 'Connected' : 'Disconnected';
    }

    function setGPUBadge(name) {
        if (name) {
            dom.gpuBadge.className = 'status-badge online';
            dom.gpuBadgeText.textContent = name;
        } else {
            dom.gpuBadge.className = 'status-badge offline';
            dom.gpuBadgeText.textContent = 'GPU Offline';
        }
    }

    function setPipelineState(state) {
        dom.pipelineState.textContent = state.toUpperCase();
        dom.pipelineState.className = `state-chip ${state}`;

        const running = state === 'running';
        const paused = state === 'paused';
        const active = running || paused;

        dom.btnStart.disabled = active;
        dom.btnPause.disabled = !active;
        dom.btnStop.disabled = !active;

        // Update Pause/Resume button icon and label safely
        const pauseIcon = dom.btnPause.querySelector('.btn-icon-left');
        const pauseLabel = dom.btnPause.querySelector('.btn-label');
        if (pauseIcon) pauseIcon.textContent = paused ? '▶' : '⏸';
        if (pauseLabel) pauseLabel.textContent = paused ? 'Resume' : 'Pause';

        if (running && !pipelineStartTime) {
            pipelineStartTime = Date.now();
            startElapsedTimer();
            resetPhases();
        }
        if (!active) {
            pipelineStartTime = null;
            stopElapsedTimer();
        }
    }

    function updateOverall(pct) {
        const rounded = Math.round(pct * 10) / 10;
        dom.overallPct.textContent = `${Math.round(pct)}%`;
        dom.overallBar.style.width = `${rounded}%`;

        if (pct > 0 && pct < 100) {
            dom.overallBar.classList.add('active');
        } else {
            dom.overallBar.classList.remove('active');
        }

        // ETA calculation
        if (pipelineStartTime && pct > 2) {
            const elapsed = (Date.now() - pipelineStartTime) / 1000;
            const rate = pct / elapsed;
            const remaining = (100 - pct) / rate;
            dom.etaTime.textContent = `ETA: ${formatTime(remaining)}`;
        }
    }

    // --- Phase management ---
    // Maps orchestrator stage names → UI phase data-phase attributes
    const PHASE_MAP = {
        'Transcription': 'Transcription',
        'Audio Transcription': 'Transcription',
        'Video Analysis': 'Video Analysis',
        'Subject Tracking': 'Tracking',
        'Tracking': 'Tracking',
        'Short Generation': 'Editing',
        'Editing': 'Editing',
        'Metadata Generation': 'Export',
        'Thumbnail Extraction': 'Export',
        'Quality Assurance': 'Export',
        'Export': 'Export',
    };

    function getPhaseEl(stageName) {
        const mapped = PHASE_MAP[stageName] || stageName;
        return $(`.phase[data-phase="${mapped}"]`);
    }

    function setPhaseStatus(stageName, status, pct, detail) {
        const el = getPhaseEl(stageName);
        if (!el) return;

        const statusEl = el.querySelector('.phase-status');
        const pctEl = el.querySelector('.phase-pct');
        const barEl = el.querySelector('.progress-bar');
        const detailEl = el.querySelector('.phase-detail');

        statusEl.textContent = status.charAt(0).toUpperCase() + status.slice(1);
        statusEl.className = `phase-status ${status}`;
        pctEl.textContent = `${Math.round(pct || 0)}%`;
        barEl.style.width = `${pct || 0}%`;

        if (status === 'running') barEl.classList.add('active');
        else barEl.classList.remove('active');

        if (detail) detailEl.textContent = detail;
    }

    function onStageStart(stage, totalSteps) {
        setPhaseStatus(stage, 'running', 0);
    }

    function onStep(stage, step, current, total, pct) {
        setPhaseStatus(stage, 'running', pct || 0, step);
    }

    function onStageComplete(stage) {
        setPhaseStatus(stage, 'complete', 100);
    }

    function onError(stage, error) {
        setPhaseStatus(stage, 'error', 0, error);
    }

    function resetPhases() {
        $$('.phase').forEach(el => {
            el.querySelector('.phase-status').textContent = 'Pending';
            el.querySelector('.phase-status').className = 'phase-status pending';
            el.querySelector('.phase-pct').textContent = '0%';
            el.querySelector('.progress-bar').style.width = '0%';
            el.querySelector('.phase-detail').textContent = '';
        });
        dom.overallPct.textContent = '0%';
        dom.overallBar.style.width = '0%';
    }

    // --- Log ---
    function addLog(text, cls = 'step', scroll = true) {
        const line = document.createElement('div');
        line.className = `log-line ${cls}`;
        line.textContent = text;
        dom.logOutput.appendChild(line);

        // Cap at 500 lines
        while (dom.logOutput.children.length > 500) {
            dom.logOutput.removeChild(dom.logOutput.firstChild);
        }

        if (scroll) {
            dom.logOutput.scrollTop = dom.logOutput.scrollHeight;
        }
    }

    // --- Elapsed Timer ---
    function startElapsedTimer() {
        stopElapsedTimer();
        elapsedTimer = setInterval(() => {
            if (pipelineStartTime) {
                const elapsed = (Date.now() - pipelineStartTime) / 1000;
                dom.elapsedTime.textContent = `Elapsed: ${formatTime(elapsed)}`;
            }
        }, 1000);
    }

    function stopElapsedTimer() {
        if (elapsedTimer) clearInterval(elapsedTimer);
        elapsedTimer = null;
    }

    function formatTime(seconds) {
        const m = Math.floor(seconds / 60);
        const s = Math.floor(seconds % 60);
        return `${m}m ${s}s`;
    }

    // ========================================================================
    // API Calls
    // ========================================================================
    async function fetchJSON(endpoint, options = {}) {
        try {
            const res = await fetch(`${API}${endpoint}`, {
                headers: { 'Content-Type': 'application/json' },
                ...options,
            });
            if (!res.ok) {
                const err = await res.json().catch(() => ({ detail: res.statusText }));
                throw new Error(err.detail || res.statusText);
            }
            return await res.json();
        } catch (e) {
            addLog(`API Error: ${e.message}`, 'error');
            throw e;
        }
    }

    const DEFAULT_VIDEO_DIR = '/app/downloaded_videos';

    // --- Load Videos ---
    async function loadVideos() {
        try {
            const data = await fetchJSON('/videos');
            dom.videoSelect.innerHTML = '';

            // Default: process all videos in downloaded_videos (batch mode)
            const defaultOpt = document.createElement('option');
            defaultOpt.value = DEFAULT_VIDEO_DIR;
            const totalVids = (data.videos || []).length;
            defaultOpt.textContent = `📁 All Downloaded Videos (${totalVids} file${totalVids !== 1 ? 's' : ''})`;
            defaultOpt.selected = true;
            dom.videoSelect.appendChild(defaultOpt);

            // Individual videos
            if (data.videos && data.videos.length > 0) {
                const group = document.createElement('optgroup');
                group.label = '🎬 Individual Videos';
                for (const vid of data.videos) {
                    const opt = document.createElement('option');
                    opt.value = vid.path;
                    opt.textContent = `${vid.name} (${vid.size_mb} MB)`;
                    group.appendChild(opt);
                }
                dom.videoSelect.appendChild(group);
            }

            // Subdirectories with videos
            if (data.directories && data.directories.length > 0) {
                const group = document.createElement('optgroup');
                group.label = '📁 Subdirectories (Batch Mode)';
                for (const dir of data.directories) {
                    const opt = document.createElement('option');
                    opt.value = dir.path;
                    opt.textContent = `${dir.name} (${dir.video_count} videos)`;
                    group.appendChild(opt);
                }
                dom.videoSelect.appendChild(group);
            }
        } catch (e) {
            // Silently fail — server might not be ready yet
        }
    }

    // --- Load Outputs ---
    async function loadOutputs() {
        try {
            const data = await fetchJSON('/outputs');

            if (!data.outputs || data.outputs.length === 0) {
                dom.outputsList.innerHTML = '<div class="empty-state">No outputs yet. Run the pipeline to generate shorts.</div>';
                return;
            }

            dom.outputsList.innerHTML = '';

            for (const output of data.outputs) {
                const group = document.createElement('div');
                group.className = 'output-group';

                const header = document.createElement('div');
                header.className = 'output-group-header';
                header.innerHTML = `
                    <span class="output-group-title">${output.name}</span>
                    <span class="output-group-count">${output.num_shorts} short${output.num_shorts !== 1 ? 's' : ''}</span>
                `;

                const shortsDiv = document.createElement('div');
                shortsDiv.className = 'output-shorts';

                for (const short of output.shorts) {
                    const shortEl = document.createElement('div');
                    shortEl.className = 'output-short';
                    shortEl.innerHTML = `
                        <span class="short-name">${short.name}</span>
                        <span class="short-badge ${short.has_video ? 'has-video' : 'no-video'}">
                            ${short.has_video ? '● Video' : '○ No Video'}
                        </span>
                    `;
                    shortsDiv.appendChild(shortEl);
                }

                header.addEventListener('click', () => {
                    shortsDiv.classList.toggle('expanded');
                });

                group.appendChild(header);
                group.appendChild(shortsDiv);
                dom.outputsList.appendChild(group);
            }
        } catch (e) {
            // Silently fail
        }
    }

    // --- System Stats ---
    async function pollSystemStats() {
        try {
            const stats = await fetchJSON('/system');

            if (stats.gpu) {
                setGPUBadge(stats.gpu.name);
                dom.gpuUtilBar.style.width = `${stats.gpu.utilization_pct}%`;
                dom.gpuUtilText.textContent = `${stats.gpu.utilization_pct}%`;
                dom.vramBar.style.width = `${stats.gpu.vram_pct}%`;
                dom.vramText.textContent = `${stats.gpu.vram_used_mb}/${stats.gpu.vram_total_mb} MB`;
                dom.gpuTempText.textContent = `${stats.gpu.temp_c}°C`;

                // Color temperature based on danger
                if (stats.gpu.temp_c > 85) dom.gpuTempText.style.color = '#ef4444';
                else if (stats.gpu.temp_c > 70) dom.gpuTempText.style.color = '#f59e0b';
                else dom.gpuTempText.style.color = '#10b981';
            } else {
                setGPUBadge(null);
            }

            if (stats.ram_pct !== null) {
                dom.ramBar.style.width = `${stats.ram_pct}%`;
                dom.ramText.textContent = `${stats.ram_used_gb}/${stats.ram_total_gb} GB`;
            }
        } catch (e) {
            // Server not reachable
        }
    }

    // ========================================================================
    // Event Handlers
    // ========================================================================
    function bindEvents() {
        // Start Pipeline
        dom.btnStart.addEventListener('click', async () => {
            const videoPath = dom.customPath.value.trim() || dom.videoSelect.value || DEFAULT_VIDEO_DIR;
            if (!videoPath) {
                addLog('Please select a video or enter a path', 'error');
                return;
            }

            try {
                resetPhases();
                dom.logOutput.innerHTML = '';
                addLog(`Starting pipeline: ${videoPath}`, 'stage');

                await fetchJSON('/run', {
                    method: 'POST',
                    body: JSON.stringify({
                        video_path: videoPath,
                        num_shorts: parseInt(dom.numShorts.value) || 4,
                        use_captions: dom.toggleCaptions.checked,
                        use_effects: dom.toggleEffects.checked,
                        use_tracking: dom.toggleTracking.checked,
                        use_analysis: dom.toggleAnalysis.checked,
                    }),
                });
            } catch (e) {
                addLog(`Failed to start: ${e.message}`, 'error');
            }
        });

        // Pause
        dom.btnPause.addEventListener('click', async () => {
            try {
                const result = await fetchJSON('/pause', { method: 'POST' });
                addLog(`Pipeline ${result.status}`, 'system');
            } catch (e) {
                addLog(`Pause failed: ${e.message}`, 'error');
            }
        });

        // Kill Switch
        dom.btnStop.addEventListener('click', async () => {
            if (!confirm('Stop the pipeline immediately? This will cancel all in-progress work.')) return;
            try {
                await fetchJSON('/stop', { method: 'POST' });
                addLog('Pipeline killed!', 'error');
            } catch (e) {
                addLog(`Stop failed: ${e.message}`, 'error');
            }
        });

        // Refresh buttons
        dom.btnRefreshVideos.addEventListener('click', loadVideos);
        dom.btnRefreshOutputs.addEventListener('click', loadOutputs);

        // Clear log
        dom.btnClearLog.addEventListener('click', () => {
            dom.logOutput.innerHTML = '';
            addLog('Log cleared', 'system');
        });
    }

    // ========================================================================
    // Init
    // ========================================================================
    function init() {
        bindEvents();
        connectWS();
        loadVideos();
        loadOutputs();

        // Start system stats polling
        pollSystemStats();
        systemPollTimer = setInterval(pollSystemStats, SYSTEM_POLL_MS);

        addLog('Dashboard initialized', 'system');
    }

    // Start when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }

})();
