/**
 * Trileaf — Dashboard WebSocket client
 *
 * WebSocket message types received from server
 * --------------------------------------------
 * run_start            { run_id, total_chunks, weights }
 * chunk_start          { chunk_idx, total, text }
 * chunk_baseline       { chunk_idx, ai_score, length }
 * ensemble_candidates  { chunk_idx, orig_ai, candidates:[{style,ai_score,sem_score,...}] }
 * chunk_stage          { chunk_idx, stage, state, message }
 * chunk_log            { chunk_idx, level, message }
 * pareto_selection     { chunk_idx, pareto_front, selected_style, selected_utility }
 * chunk_done           { chunk_idx, final_text, original_ai_score,
 *                        final_ai_score, final_sem_score,
 *                        reverted_to_original, status_label, selected_style, candidates }
 * final_scoring        { message }
 * run_done             { run_id, output, original_ai_score, final_ai_score, final_sem_score,
 *                        chunks:[...] }
 * error                { message }
 */

(function () {
  'use strict';

  /* ── DOM refs ──────────────────────────────────────────────────────────── */
  const $ = id => document.getElementById(id);

  const inputText          = $('input-text');
  const runBtn             = $('run-btn');
  const modeShortBtn       = $('mode-short-btn');
  const modeLongBtn        = $('mode-long-btn');
  const modeHint           = $('mode-hint');
  const runSingleBtn       = $('run-single-btn');
  const runDoubleBtn       = $('run-double-btn');
  const runModeHint        = $('run-mode-hint');
  const chunkList          = $('chunk-list');
  const progressPH         = $('progress-placeholder');
  const runSummary         = $('run-summary');
  const outputView         = $('output-view');
  const outputStatus       = $('output-status');
  const outputStatusLabel  = $('output-status-label');
  const outputStatusDetail = $('output-status-detail');
  const viewChunkBtn       = $('view-chunk-btn');
  const viewPlainBtn       = $('view-plain-btn');
  const copyBtn            = $('copy-btn');
  const wsDot              = $('ws-dot');
  const wsLabel            = $('ws-label');
  const deviceLabel        = $('device-label');
  const rewriteLabel       = $('rewrite-label');

  /* ── State ─────────────────────────────────────────────────────────────── */
  let ws          = null;
  let running     = false;
  const cards     = {};   // chunk_idx → { el, logCount }
  const originalTexts = {}; // chunk_idx → original text (from chunk_start)
  let totalChunks = 0;
  let doneChunks  = 0;
  let outputChunks = [];
  let outputRenderMode = 'chunk';
  let liveUiTimer = null;
  let overallVisualPct = 0;

  // Chunk mode: "short" | "long"
  let chunkMode = 'short';

  // Run mode: "single" | "double"
  let runMode = 'single';

  // Double-pass state — null when not in a two-pass run.
  // { pass: 1|2, firstPassOriginalAiScore: number|null, chunkMode: string }
  let _doublePassState = null;

  const CHUNK_PROGRESS = {
    boot: 4,
    rewriteStart: 10,
    rewriteSpan: 50,
    batchStart: 66,
    batchSpan: 20,
    paretoStart: 90,
    paretoSpan: 7,
  };

  function createOutputChunkEntry(chunkIdx) {
    return {
      chunk_idx:            chunkIdx,
      para_idx:             chunkIdx,  // default: each chunk is its own para; overwritten by server
      original_text:        '',
      output_text:          '',
      reverted_to_original: false,
      best_gated:           null,
      status_label:         'Queued',
      original_ai_score:    0,
      final_ai_score:       0,
      final_sem_score:      0,
      selected_style:       '',
      mode:                 'output',
      is_complete:          false,
    };
  }

  function upsertOutputChunk(chunkIdx, patch = {}) {
    let chunk = outputChunks.find(item => item.chunk_idx === chunkIdx);
    if (!chunk) {
      chunk = createOutputChunkEntry(chunkIdx);
      outputChunks.push(chunk);
    }

    Object.assign(chunk, patch);
    outputChunks.sort((a, b) => a.chunk_idx - b.chunk_idx);
    return chunk;
  }

  function resolveOutputChunkMode(existingChunk, revertedToOriginal) {
    if (!revertedToOriginal) return existingChunk?.mode || 'output';
    if (existingChunk?.is_complete) return existingChunk.mode || 'original';
    return 'original';
  }

  /* ── Copy button ───────────────────────────────────────────────────────── */
  copyBtn.addEventListener('click', () => {
    const t = assembleOutputText();
    if (!t) return;
    navigator.clipboard.writeText(t).catch(() => {
      const temp = document.createElement('textarea');
      temp.value = t;
      document.body.appendChild(temp);
      temp.select();
      document.execCommand('copy');
      temp.remove();
    });
    copyBtn.textContent = 'Copied!';
    setTimeout(() => { copyBtn.textContent = 'Copy'; }, 1500);
  });

  viewChunkBtn.addEventListener('click', () => {
    outputRenderMode = 'chunk';
    syncOutputViewButtons();
    renderOutputView();
  });

  viewPlainBtn.addEventListener('click', () => {
    outputRenderMode = 'plain';
    syncOutputViewButtons();
    renderOutputView();
  });

  /* ── Mode toggle ───────────────────────────────────────────────────────── */
  const _modeHints = {
    short: 'Fine-grained · ~200-char chunks · best for texts up to ~3 000 chars',
    long:  'Paragraph-aware · ~400-char chunks · recommended for 2 000 – 8 000 chars',
  };

  function setChunkMode(mode) {
    chunkMode = mode;
    modeShortBtn.classList.toggle('active', mode === 'short');
    modeLongBtn.classList.toggle('active',  mode === 'long');
    if (modeHint) modeHint.textContent = _modeHints[mode];
  }

  if (modeShortBtn) modeShortBtn.addEventListener('click', () => setChunkMode('short'));
  if (modeLongBtn)  modeLongBtn.addEventListener('click',  () => setChunkMode('long'));

  /* ── Run mode toggle ───────────────────────────────────────────────────── */
  const _runModeHints = {
    single: 'One standard optimization pass',
    double: 'Text passes through the pipeline twice · may lower AI score further · doubles processing time · greater risk of semantic drift',
  };

  function setRunMode(mode) {
    runMode = mode;
    if (runSingleBtn) runSingleBtn.classList.toggle('active', mode === 'single');
    if (runDoubleBtn) runDoubleBtn.classList.toggle('active', mode === 'double');
    if (runModeHint)  runModeHint.textContent = _runModeHints[mode];
  }

  if (runSingleBtn) runSingleBtn.addEventListener('click', () => setRunMode('single'));
  if (runDoubleBtn) runDoubleBtn.addEventListener('click', () => setRunMode('double'));

  /* ── Fetch health info ─────────────────────────────────────────────────── */
  function fetchHealth() {
    fetch('/api/health')
      .then(r => r.json())
      .then(d => {
        deviceLabel.textContent = d.device || '?';
        const backend = d.rewrite_backend || '?';
        const model   = d.rewrite_model  || '';
        const profile = d.rewrite_profile || '';
        let label = model || backend;
        if (profile) label += ` (${profile})`;
        rewriteLabel.textContent = label;
      })
      .catch(() => {});
  }

  /* ── WebSocket ─────────────────────────────────────────────────────────── */
  let _reconnectScheduled = false;

  function _scheduleReconnect() {
    if (_reconnectScheduled) return;
    _reconnectScheduled = true;
    setTimeout(() => { _reconnectScheduled = false; connect(); }, 3000);
  }

  function connect() {
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    ws = new WebSocket(`${proto}://${location.host}/ws/optimizer`);

    ws.onopen = () => {
      wsDot.className     = 'ws-dot connected';
      wsLabel.textContent = 'Connected';
      // If the server restarted while a run was in progress, the run is gone.
      // Reset running state so the button becomes usable again.
      if (running) setRunning(false);
      fetchHealth();
      // Re-fetch after warm-up is expected to complete (~5 s import + buffer)
      // so Device / Rewrite labels show the fully-resolved values.
      setTimeout(fetchHealth, 8000);
    };

    ws.onclose = () => {
      wsDot.className     = 'ws-dot';
      wsLabel.textContent = 'Disconnected';
      _scheduleReconnect();
    };

    ws.onerror = () => {
      wsDot.className     = 'ws-dot';
      wsLabel.textContent = 'Error';
      _scheduleReconnect();
    };

    ws.onmessage = evt => {
      let msg;
      try { msg = JSON.parse(evt.data); } catch { return; }
      dispatch(msg);
    };
  }

  /* ── Message dispatcher ────────────────────────────────────────────────── */
  function dispatch({ type, data }) {
    switch (type) {
      case 'run_start':           onRunStart(data);           break;
      case 'chunk_start':         onChunkStart(data);         break;
      case 'chunk_baseline':      /* log only — handled via chunk_log */  break;
      case 'rewrite_candidate':   onRewriteCandidate(data);   break;
      case 'ensemble_candidates': onEnsembleCandidates(data); break;
      case 'chunk_stage':         onChunkStage(data);         break;
      case 'chunk_log':           onChunkLog(data);           break;
      case 'pareto_selection':    onParetoSelection(data);    break;
      case 'chunk_done':          onChunkDone(data);          break;
      case 'final_scoring':       onFinalScoring(data);       break;
      case 'run_done':            onRunDone(data);            break;
      case 'error':               onError(data);              break;
    }
  }

  /* ── Event handlers ────────────────────────────────────────────────────── */

  /**
   * Reset the UI and immediately render a chunk-0 placeholder.
   * Called on button click so feedback appears before any WS event arrives.
   * totalChunks is set to 0 here; onRunStart updates it to the real value.
   */
  function _prepareRunUI() {
    Object.keys(cards).forEach(k => delete cards[k]);
    Object.keys(originalTexts).forEach(k => delete originalTexts[k]);
    chunkList.innerHTML = '';
    if (progressPH) {
      chunkList.appendChild(progressPH);
      progressPH.style.display = 'none';
    }

    outputRenderMode = 'chunk';
    outputChunks = [];
    syncOutputViewButtons();
    renderOutputView();
    clearOutputStatus();

    const ob = $('overall-bar');
    if (ob) ob.style.width = '0%';
    overallVisualPct = 0;
    totalChunks = 0;
    doneChunks  = 0;
    runSummary.textContent = 'Starting…';

    // Chunk 0 placeholder — title shows "Chunk 1 / ?" until run_start arrives
    const card = buildCard(0, '?');
    cards[0] = card;
    chunkList.appendChild(card.el);

    setRunning(true);
  }

  function onRunStart(data) {
    // UI and chunk-0 placeholder already set up by _prepareRunUI() on click.
    // Just sync the now-known chunk count and update the placeholder title.
    totalChunks = data.total_chunks;
    doneChunks  = 0;
    runSummary.textContent = `0 / ${totalChunks} chunks done`;

    const card0 = cards[0];
    if (card0) {
      const titleEl = card0.el.querySelector('.chunk-title');
      if (titleEl) titleEl.textContent = `Chunk 1 / ${totalChunks}`;
    } else {
      // Defensive: button click didn't run (e.g. direct WS trigger) — create now
      const card = buildCard(0, totalChunks);
      cards[0] = card;
      chunkList.appendChild(card.el);
    }
  }

  function onChunkStart(data) {
    originalTexts[data.chunk_idx] = data.text;

    // Reuse the pre-rendered placeholder if it exists, otherwise create a new card
    let card = cards[data.chunk_idx];
    if (!card) {
      card = buildCard(data.chunk_idx, data.total);
      cards[data.chunk_idx] = card;
      chunkList.appendChild(card.el);
    }
    card.el.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
    updateCardLiveState(data.chunk_idx, {
      stage: 'boot',
      detailBase: 'Preparing rewrite stage',
      statusBase: 'Preparing…',
      summaryBase: `Chunk ${data.chunk_idx + 1}/${data.total} · preparing rewrite stage`,
      startedAt: Date.now(),
    });
    setCardProgressFloor(data.chunk_idx, CHUNK_PROGRESS.boot);
    setDraftProgressFloor(data.chunk_idx, 0);
    setConnProgress(data.chunk_idx, 'rewrite', 0);
    setStage(data.chunk_idx, 'rewrite', 'active');
    setPhase(data.chunk_idx, 'Ensemble', 'Preparing drafts…');
    setStatus(data.chunk_idx, 'Rewriting…');
  }

  function onRewriteCandidate(data) {
    const { chunk_idx, style, style_idx, total_styles, status } = data;
    const n         = style_idx + 1;
    const label     = style.charAt(0).toUpperCase() + style.slice(1);
    const icon      = status === 'done' ? '✓' : status === 'error' ? '✗' : '…';
    const subText   = `${icon} ${n}/${total_styles} · ${label}`;
    const chunkLabel = totalChunks ? `Chunk ${chunk_idx + 1}/${totalChunks}` : `Chunk ${chunk_idx + 1}`;
    const draftBase = `Draft ${n}/${total_styles} · ${label}`;
    const draftStartPct = CHUNK_PROGRESS.rewriteStart + ((n - 1) * CHUNK_PROGRESS.rewriteSpan / total_styles);
    const draftDonePct = CHUNK_PROGRESS.rewriteStart + (n * CHUNK_PROGRESS.rewriteSpan / total_styles);
    const ensembleStartPct = ((n - 1) / total_styles) * 100;
    const ensembleDonePct = (n / total_styles) * 100;

    if (status === 'generating') {
      setCardProgressFloor(chunk_idx, draftStartPct);
      setDraftProgressFloor(chunk_idx, ensembleStartPct);
      updateCardLiveState(chunk_idx, {
        stage: 'rewrite',
        draftIndex: style_idx,
        totalDrafts: total_styles,
        label,
        segmentStart: ensembleStartPct,
        segmentEnd: ensembleDonePct,
        detailBase: `Generating ${draftBase}`,
        statusBase: `Draft ${n}/${total_styles}`,
        summaryBase: `${chunkLabel} · generating ${draftBase.toLowerCase()}`,
        startedAt: Date.now(),
      });
      setPhase(chunk_idx, 'Ensemble', `Generating ${draftBase}`);
      setStepSub(chunk_idx, 'rewrite', `${n}/${total_styles} · ${label}`);
      setStatus(chunk_idx, `Draft ${n}/${total_styles}…`);
    } else {
      setCardProgressFloor(chunk_idx, draftDonePct);
      setDraftProgressFloor(chunk_idx, ensembleDonePct);
      // done or error — briefly show result then the next "generating" will overwrite
      setStepSub(chunk_idx, 'rewrite', subText);
      setStatus(chunk_idx, status === 'error' ? `Draft ${n}/${total_styles} fallback` : `Draft ${n}/${total_styles} ready`);
      if (n === total_styles) {
        // last candidate finished — summarise
        setConnProgress(chunk_idx, 'rewrite', 100);
        setPhase(chunk_idx, 'Ensemble', `All ${total_styles} drafts generated`);
        updateCardLiveState(chunk_idx, {
          stage: 'rewrite-wrap',
          detailBase: `All ${total_styles} drafts generated`,
          statusBase: 'Preparing comparison',
          summaryBase: `${chunkLabel} · drafts ready, preparing comparison`,
          startedAt: Date.now(),
        });
      }
    }
  }

  function onEnsembleCandidates(data) {
    // data.candidates: [{style, ai_score, sem_score, ...}]
    const scores = (data.candidates || [])
      .map(c => `${c.style[0].toUpperCase()}: ${fmtPct(c.ai_score)}`)
      .join(' · ');
    const best = (data.candidates || []).reduce(
      (a, b) => (!a || b.ai_score < a.ai_score) ? b : a, null
    );
    setCardProgressFloor(data.chunk_idx, CHUNK_PROGRESS.batchStart - 3);
    if (best) setStepSub(data.chunk_idx, 'rewrite', `${best.style} ${fmtPct(best.ai_score)}`);
    setPhase(data.chunk_idx, 'Ensemble scored', scores);
    setStatus(data.chunk_idx, 'Batch scoring candidates…');
  }

  function onChunkStage(data) {
    const labelMap = {
      rewrite:     'Ensemble',
      batch_score: 'Batch Score',
      pareto:      'Pareto',
    };
    const tone = data.state === 'warn' ? 'warn' : data.state === 'done' ? 'done' : '';
    setPhase(data.chunk_idx, labelMap[data.stage] || 'Working', data.message, tone);

    if (data.stage === 'rewrite') {
      setStage(data.chunk_idx, 'rewrite', data.state);
      if (data.state === 'done' || data.state === 'warn') {
        setCardProgressFloor(data.chunk_idx, CHUNK_PROGRESS.rewriteStart + CHUNK_PROGRESS.rewriteSpan);
        setConnProgress(data.chunk_idx, 'rewrite', 100);
        const conn = getConn(data.chunk_idx, 'rewrite');
        if (conn) conn.classList.add('done');
      }
    }
    if (data.stage === 'batch_score') {
      if (data.state === 'active') {
        const chunkLabel = totalChunks ? `Chunk ${data.chunk_idx + 1}/${totalChunks}` : `Chunk ${data.chunk_idx + 1}`;
        setCardProgressFloor(data.chunk_idx, CHUNK_PROGRESS.batchStart);
        setConnProgress(data.chunk_idx, 'rewrite', 100);
        updateCardLiveState(data.chunk_idx, {
          stage: 'batch',
          detailBase: 'Comparing drafts and scoring candidates',
          statusBase: 'Scoring drafts',
          summaryBase: `${chunkLabel} · comparing drafts and scoring`,
          startedAt: Date.now(),
        });
      } else if (data.state === 'done' || data.state === 'warn') {
        setCardProgressFloor(data.chunk_idx, CHUNK_PROGRESS.batchStart + CHUNK_PROGRESS.batchSpan);
      }
      setStage(data.chunk_idx, 'batch_score', data.state);
      if (data.state === 'done' || data.state === 'warn') {
        const conn = getConn(data.chunk_idx, 'batch_score');
        if (conn) conn.classList.add('done');
      }
    }
    if (data.stage === 'pareto') {
      if (data.state === 'active') {
        const chunkLabel = totalChunks ? `Chunk ${data.chunk_idx + 1}/${totalChunks}` : `Chunk ${data.chunk_idx + 1}`;
        setCardProgressFloor(data.chunk_idx, CHUNK_PROGRESS.paretoStart);
        updateCardLiveState(data.chunk_idx, {
          stage: 'pareto',
          detailBase: 'Selecting the strongest rewrite',
          statusBase: 'Selecting best draft',
          summaryBase: `${chunkLabel} · selecting the strongest rewrite`,
          startedAt: Date.now(),
        });
      } else if (data.state === 'done' || data.state === 'warn') {
        setCardProgressFloor(data.chunk_idx, CHUNK_PROGRESS.paretoStart + CHUNK_PROGRESS.paretoSpan);
      }
      setStage(data.chunk_idx, 'pareto', data.state);
    }
  }

  function onChunkLog(data) {
    pushLog(data.chunk_idx, data.message, data.level || 'info');
  }

  function onParetoSelection(data) {
    setStepSub(
      data.chunk_idx, 'pareto',
      data.selected_style
        ? `${data.selected_style} U=${data.selected_utility?.toFixed(2)}`
        : 'fallback'
    );
  }

  function onChunkDone(data) {
    setStage(data.chunk_idx, 'pareto', data.reverted_to_original ? 'warn' : 'done');
    setCardProgressFloor(data.chunk_idx, 100);
    setDraftProgressFloor(data.chunk_idx, 100);
    clearCardLiveState(data.chunk_idx);
    setConnProgress(data.chunk_idx, 'rewrite', 100);

    const card = cards[data.chunk_idx];
    if (card) {
      card.completed = true;
      card.el.classList.remove('active-card');
      card.el.classList.remove('processing-live', 'inference-live');
      card.el.classList.add(data.reverted_to_original ? 'warn' : 'done');
    }

    const gatePassCount = (data.candidates || []).filter(c => c.gate_pass).length;
    const detailMsg = data.reverted_to_original
      ? `No candidate passed gate (${gatePassCount}/3 passed)`
      : `${data.selected_style} · AI ${fmtPct(data.final_ai_score)} · Sem ${fmtPct(data.final_sem_score)}`;

    setPhase(
      data.chunk_idx,
      data.status_label || (data.reverted_to_original ? 'Reverted to original' : 'Edited'),
      detailMsg,
      data.reverted_to_original ? 'warn' : 'done'
    );
    setStatus(data.chunk_idx, data.reverted_to_original ? 'Fallback ⚠' : 'Done ✓');

    doneChunks++;
    runSummary.textContent = `${doneChunks} / ${totalChunks} chunks done`;
    overallVisualPct = Math.max(
      overallVisualPct,
      (doneChunks / Math.max(totalChunks, 1)) * 100
    );

    const ob = $('overall-bar');
    if (ob) ob.style.width = (doneChunks / totalChunks * 100).toFixed(1) + '%';

    const existingOutputChunk = outputChunks.find(chunk => chunk.chunk_idx === data.chunk_idx);

    // In pass 2 of a double run, show the raw-input chunk as "original" so the
    // per-chunk comparison reflects raw-input → final-output, not pass1-output → pass2-output.
    const _p1raw = _doublePassState?.pass === 2
      ? (_doublePassState.pass1RawChunks?.[data.chunk_idx] ?? null)
      : null;

    upsertOutputChunk(data.chunk_idx, {
      para_idx:             data.para_idx ?? data.chunk_idx,
      original_text:        _p1raw ? _p1raw.text : (originalTexts[data.chunk_idx] || existingOutputChunk?.original_text || ''),
      output_text:          data.final_text || '',
      reverted_to_original: !!data.reverted_to_original,
      best_gated:           data.best_candidate || existingOutputChunk?.best_gated || null,
      status_label:         data.status_label || (data.reverted_to_original ? 'Reverted to original' : 'Edited'),
      original_ai_score:    _p1raw ? _p1raw.original_ai_score : (data.original_ai_score || 0),
      final_ai_score:       data.final_ai_score || 0,
      final_sem_score:      data.final_sem_score || 0,
      selected_style:       data.selected_style || '',
      mode:                 resolveOutputChunkMode(existingOutputChunk, !!data.reverted_to_original),
      is_complete:          true,
    });
    renderOutputView();
  }

  function onFinalScoring() {
    runSummary.textContent = 'Computing final scores…';
    overallVisualPct = 100;
    const ob  = $('overall-bar');
    const opr = ob && ob.parentElement;
    if (ob)  { ob.style.width = '100%'; ob.classList.remove('running'); }
    if (opr) opr.classList.remove('running');
  }

  function onRunDone(data) {
    // Patch existing streamed chunks with best_candidate (text only available here)
    (data.chunks || []).forEach(chunk => {
      const existing = outputChunks.find(c => c.chunk_idx === chunk.chunk_idx);
      const _p1raw = _doublePassState?.pass === 2
        ? (_doublePassState.pass1RawChunks?.[chunk.chunk_idx] ?? null)
        : null;
      upsertOutputChunk(chunk.chunk_idx, {
        para_idx:             chunk.para_idx ?? chunk.chunk_idx,
        original_text:        _p1raw ? _p1raw.text : (chunk.original_text || existing?.original_text || ''),
        output_text:          chunk.final_text || existing?.output_text || '',
        reverted_to_original: !!chunk.reverted_to_original,
        best_gated:           chunk.best_candidate || null,
        status_label:         chunk.status_label || (chunk.reverted_to_original ? 'Reverted to original' : 'Edited'),
        original_ai_score:    _p1raw ? _p1raw.original_ai_score : (chunk.original_ai_score || existing?.original_ai_score || 0),
        final_ai_score:       chunk.final_ai_score || existing?.final_ai_score || 0,
        final_sem_score:      chunk.final_sem_score || existing?.final_sem_score || 0,
        selected_style:       chunk.selected_style || existing?.selected_style || '',
        mode:                 resolveOutputChunkMode(existing, !!chunk.reverted_to_original),
        is_complete:          true,
      });
    });
    renderOutputView();

    // ── Two-pass: after pass 1, fire pass 2 automatically ──────────────────
    if (_doublePassState && _doublePassState.pass === 1) {
      _doublePassState.firstPassOriginalAiScore = data.original_ai_score || 0;
      // Snapshot per-chunk raw-input data BEFORE _prepareRunUI() clears outputChunks.
      // Pass 2 will use these to show raw-input vs final-output in each chunk card.
      _doublePassState.pass1RawChunks = {};
      outputChunks.forEach(c => {
        _doublePassState.pass1RawChunks[c.chunk_idx] = {
          text:              c.original_text,
          original_ai_score: c.original_ai_score,
        };
      });
      _doublePassState.pass = 2;

      const secondPassText = data.output;
      _prepareRunUI();
      if (runSummary) runSummary.textContent = 'Pass 2 / 2';

      fetch('/api/optimize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: secondPassText, chunk_mode: _doublePassState.chunkMode }),
      }).then(r => r.json()).then(d => {
        if (d.error) { alert(d.error); setRunning(false); _doublePassState = null; }
      }).catch(e => {
        alert('Pass 2 failed: ' + e.message);
        setRunning(false);
        _doublePassState = null;
      });
      return; // defer UI finalisation to pass-2 run_done
    }

    // ── Finalise UI (single pass, or pass 2 of double pass) ────────────────
    const revertedCount = (data.chunks || []).filter(c => c.reverted_to_original).length;

    // For double-pass, compare against the very first original AI score
    const origAi  = _doublePassState
      ? _doublePassState.firstPassOriginalAiScore
      : (data.original_ai_score || 0);
    const finalAi = data.final_ai_score || 0;

    const delta    = (origAi - finalAi) * 100;
    const deltaStr = delta >= 0
      ? `<span style="color:var(--green)">↓${delta.toFixed(1)}pp</span>`
      : `<span style="color:var(--red)">↑${Math.abs(delta).toFixed(1)}pp</span>`;
    const passLabel = _doublePassState
      ? `<span style="color:var(--muted);font-size:10px"> (2-pass)</span>`
      : '';
    const aiCompare = `<span style="color:var(--muted)">AI detection &nbsp;</span>`
      + `<span style="font-weight:700;color:#ffb0b0">${fmtPct(origAi)}</span>`
      + `<span style="color:var(--muted)"> before &nbsp;→&nbsp; </span>`
      + `<span style="font-weight:700;color:var(--green)">${fmtPct(finalAi)}</span>`
      + `<span style="color:var(--muted)"> after &nbsp;</span>`
      + deltaStr + passLabel;
    const semStr = `Semantic similarity: ${fmtPct(data.final_sem_score)}`;

    if (revertedCount > 0) {
      setOutputStatus(
        'warn',
        aiCompare,
        `${semStr}  ·  ${revertedCount} chunk${revertedCount > 1 ? 's' : ''} reverted to original`
      );
      runSummary.textContent = `Done — ${revertedCount} chunk${revertedCount > 1 ? 's' : ''} reverted`;
    } else {
      setOutputStatus('done', aiCompare, semStr);
      runSummary.textContent = `Done — ${totalChunks} chunks`;
    }

    _doublePassState = null;
    setRunning(false);
  }

  function onError(data) {
    setRunning(false);
    const err = document.createElement('div');
    err.className = 'chunk-card error-card';
    err.innerHTML = `<div style="color:var(--red);font-size:12px;">
      ⚠ Error: ${escHtml(data.message)}
    </div>`;
    chunkList.appendChild(err);
    err.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  }

  /* ── Run button ─────────────────────────────────────────────────────────── */
  runBtn.addEventListener('click', async () => {
    const text = inputText.value.trim();
    if (!text) { inputText.focus(); return; }

    if (runMode === 'double') {
      _doublePassState = { pass: 1, firstPassOriginalAiScore: null, chunkMode };
      _prepareRunUI();
      if (runSummary) runSummary.textContent = 'Pass 1 / 2';
    } else {
      _doublePassState = null;
      _prepareRunUI();
    }

    try {
      const res = await fetch('/api/optimize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, chunk_mode: chunkMode }),
      });
      const data = await res.json();
      if (data.error) { alert(data.error); setRunning(false); _doublePassState = null; }
    } catch (e) {
      alert('Failed to start optimization: ' + e.message);
      setRunning(false);
      _doublePassState = null;
    }
  });

  /* ── Card helpers ──────────────────────────────────────────────────────── */

  function buildCard(idx, total) {
    const el = document.createElement('div');
    el.className = 'chunk-card active-card';
    el.dataset.idx = idx;
    el.innerHTML = `
      <div class="chunk-head">
        <span class="chunk-title">Chunk ${idx + 1} / ${total}</span>
        <span class="chunk-status" data-status="${idx}">Queued…</span>
      </div>
      <div class="chunk-meta">
        <div class="phase-badge" data-phase-label="${idx}">Queued</div>
        <div class="phase-detail" data-phase-detail="${idx}">Waiting to start</div>
      </div>
      <div class="pipeline-steps">
        <div class="step" data-stage="rewrite" data-idx="${idx}">
          <div class="step-dot-wrap"><div class="step-dot"></div></div>
          <div class="step-label">Ensemble</div>
          <div class="step-sub" data-sub="rewrite" data-idx="${idx}"></div>
        </div>
        <div class="step-conn" data-conn="rewrite" data-idx="${idx}"></div>
        <div class="step" data-stage="batch_score" data-idx="${idx}">
          <div class="step-dot-wrap"><div class="step-dot"></div></div>
          <div class="step-label">Batch Score</div>
          <div class="step-sub" data-sub="batch_score" data-idx="${idx}"></div>
        </div>
        <div class="step-conn" data-conn="batch_score" data-idx="${idx}"></div>
        <div class="step" data-stage="pareto" data-idx="${idx}">
          <div class="step-dot-wrap"><div class="step-dot"></div></div>
          <div class="step-label">Pareto</div>
          <div class="step-sub" data-sub="pareto" data-idx="${idx}"></div>
        </div>
      </div>
      <div class="chunk-log">
        <div class="chunk-log-head">
          <span>Recent Activity</span>
          <span data-log-count="${idx}">0</span>
        </div>
        <div class="chunk-log-list" data-log-list="${idx}"></div>
      </div>`;
    return {
      el,
      idx,
      logCount: 0,
      completed: false,
      progressFloor: 0,
      draftProgressFloor: 0,
      live: null,
    };
  }

  function setStage(idx, stage, state) {
    const card = cards[idx];
    if (!card) return;
    const step = card.el.querySelector(`.step[data-stage="${stage}"][data-idx="${idx}"]`);
    if (step) step.className = `step ${state}`;
  }

  function getConn(idx, stage) {
    const card = cards[idx];
    if (!card) return null;
    return card.el.querySelector(`.step-conn[data-conn="${stage}"][data-idx="${idx}"]`);
  }

  function setPhase(idx, label, detail, tone = '') {
    const card = cards[idx];
    if (!card) return;
    const labelEl  = card.el.querySelector(`[data-phase-label="${idx}"]`);
    const detailEl = card.el.querySelector(`[data-phase-detail="${idx}"]`);
    if (labelEl) {
      labelEl.textContent = label;
      labelEl.className = `phase-badge${tone ? ' ' + tone : ''}${card.live ? ' live' : ''}`;
    }
    if (detailEl) detailEl.textContent = detail;
    if (card.el && tone !== 'done') card.el.classList.add('active-card');
  }

  function setStepSub(idx, stage, text) {
    const card = cards[idx];
    if (!card) return;
    const sub = card.el.querySelector(`[data-sub="${stage}"][data-idx="${idx}"]`);
    if (sub) sub.textContent = text;
  }

  function setStatus(idx, text) {
    const card = cards[idx];
    if (!card) return;
    const el = card.el.querySelector(`[data-status="${idx}"]`);
    if (el) el.textContent = text;
  }

  function updateCardLiveState(idx, live) {
    const card = cards[idx];
    if (!card) return;
    card.live = live;
    card.el.classList.toggle('processing-live', !!live);
    card.el.classList.toggle('inference-live', !!live && live.stage === 'rewrite');
    const labelEl = card.el.querySelector(`[data-phase-label="${idx}"]`);
    const detailEl = card.el.querySelector(`[data-phase-detail="${idx}"]`);
    const statusEl = card.el.querySelector(`[data-status="${idx}"]`);
    if (labelEl) labelEl.classList.toggle('live', !!live);
    if (detailEl) detailEl.classList.toggle('live', !!live);
    if (statusEl) statusEl.classList.toggle('live', !!live);
  }

  function clearCardLiveState(idx) {
    updateCardLiveState(idx, null);
  }

  function setCardProgressFloor(idx, pct) {
    const card = cards[idx];
    if (!card) return;
    card.progressFloor = Math.max(card.progressFloor || 0, Math.min(100, pct));
  }

  function setDraftProgressFloor(idx, pct) {
    const card = cards[idx];
    if (!card) return;
    card.draftProgressFloor = Math.max(card.draftProgressFloor || 0, Math.min(100, pct));
  }

  function setConnProgress(idx, stage, pct) {
    const conn = getConn(idx, stage);
    if (!conn) return;
    const clamped = Math.max(0, Math.min(100, pct));
    conn.style.setProperty('--progress', `${clamped.toFixed(1)}%`);
  }

  function getActiveCard() {
    return Object.values(cards)
      .filter(card => card && !card.completed)
      .sort((a, b) => a.idx - b.idx)[0] || null;
  }

  function estimateChunkProgress(card) {
    const live = card && card.live;
    const floor = Math.max(0, Math.min(100, card?.progressFloor || 0));
    if (!live) return floor;
    const elapsed = Math.max(0, (Date.now() - (live.startedAt || Date.now())) / 1000);
    const ease = 1 - Math.exp(-elapsed / 4.2);

    if (live.stage === 'rewrite') {
      const totalDrafts = Math.max(1, live.totalDrafts || 3);
      const draftIndex = Math.max(0, live.draftIndex || 0);
      const segStart = CHUNK_PROGRESS.rewriteStart + (draftIndex * CHUNK_PROGRESS.rewriteSpan / totalDrafts);
      const segSpan = CHUNK_PROGRESS.rewriteSpan / totalDrafts;
      const simulated = segStart + segSpan * Math.min(0.86, ease);
      return Math.max(floor, simulated);
    }
    if (live.stage === 'rewrite-wrap') {
      return Math.max(floor, CHUNK_PROGRESS.batchStart - 2);
    }
    if (live.stage === 'batch') {
      const simulated = CHUNK_PROGRESS.batchStart + CHUNK_PROGRESS.batchSpan * Math.min(0.92, ease);
      return Math.max(floor, simulated);
    }
    if (live.stage === 'pareto') {
      const simulated = CHUNK_PROGRESS.paretoStart + CHUNK_PROGRESS.paretoSpan * Math.min(0.92, ease);
      return Math.max(floor, simulated);
    }
    if (live.stage === 'boot') {
      return Math.max(floor, CHUNK_PROGRESS.boot + 2 * Math.min(1, ease));
    }
    return floor;
  }

  function estimateDraftProgress(card) {
    const live = card && card.live;
    const floor = Math.max(0, Math.min(100, card?.draftProgressFloor || 0));
    if (!live || live.stage !== 'rewrite') return floor;

    const elapsed = Math.max(0, (Date.now() - (live.startedAt || Date.now())) / 1000);
    const ease = 1 - Math.exp(-elapsed / 5.4);
    const segStart = Math.max(0, Math.min(100, live.segmentStart || 0));
    const segEnd = Math.max(segStart, Math.min(100, live.segmentEnd || 100));
    const segSpan = segEnd - segStart;
    const simulated = segStart + segSpan * Math.min(0.82, ease);
    return Math.max(floor, simulated);
  }

  function updateLiveCardText(card) {
    if (!card || !card.live) return;
    const idx = card.idx;
    const live = card.live;
    const detailEl = card.el.querySelector(`[data-phase-detail="${idx}"]`);
    const statusEl = card.el.querySelector(`[data-status="${idx}"]`);
    const summaryBase = live.summaryBase || '';

    if (detailEl && live.detailBase) detailEl.textContent = live.detailBase;
    if (statusEl && live.statusBase) statusEl.textContent = live.statusBase;
    if (summaryBase && runSummary) runSummary.textContent = summaryBase;
  }

  function updateOverallProgressFrame() {
    const ob = $('overall-bar');
    if (!ob || !totalChunks) return;
    const activeCard = getActiveCard();
    const activeChunkPct = activeCard ? (estimateChunkProgress(activeCard) / 100) : 0;
    const target = Math.max(
      overallVisualPct,
      ((doneChunks + activeChunkPct) / Math.max(totalChunks, 1)) * 100
    );
    overallVisualPct += (target - overallVisualPct) * 0.18;
    if (target - overallVisualPct < 0.2) overallVisualPct = target;
    const renderPct = activeCard ? Math.min(99.4, overallVisualPct) : overallVisualPct;
    ob.style.width = `${renderPct.toFixed(2)}%`;

    if (activeCard) {
      const draftPct = estimateDraftProgress(activeCard);
      setConnProgress(activeCard.idx, 'rewrite', draftPct);
    }
  }

  function tickLiveUi() {
    if (!running) return;
    const activeCard = getActiveCard();
    if (activeCard) updateLiveCardText(activeCard);
    updateOverallProgressFrame();
  }

  function startLiveUiLoop() {
    if (liveUiTimer) return;
    liveUiTimer = setInterval(tickLiveUi, 180);
    tickLiveUi();
  }

  function stopLiveUiLoop() {
    if (!liveUiTimer) return;
    clearInterval(liveUiTimer);
    liveUiTimer = null;
  }

  function pushLog(idx, message, level) {
    const card = cards[idx];
    if (!card) return;
    const list    = card.el.querySelector(`[data-log-list="${idx}"]`);
    const counter = card.el.querySelector(`[data-log-count="${idx}"]`);
    if (!list) return;

    const line = document.createElement('div');
    line.className = `log-line ${level || 'info'}`.trim();
    line.innerHTML = `
      <span class="log-time">${timeStamp()}</span>
      <span class="log-msg">${escHtml(message)}</span>
    `;
    list.prepend(line);

    while (list.children.length > 40) list.removeChild(list.lastChild);

    card.logCount = (card.logCount || 0) + 1;
    if (counter) counter.textContent = `${Math.min(card.logCount, 40)} shown`;
  }

  // label accepts raw HTML; detail is plain text
  function setOutputStatus(tone, labelHtml, detail) {
    if (!outputStatus) return;
    outputStatus.className = `output-status visible${tone ? ' ' + tone : ''}`;
    if (outputStatusLabel)  outputStatusLabel.innerHTML = labelHtml || '';
    if (outputStatusDetail) outputStatusDetail.textContent = detail || '';
  }

  function clearOutputStatus() {
    if (!outputStatus) return;
    outputStatus.className = 'output-status';
    if (outputStatusLabel)  outputStatusLabel.textContent  = '';
    if (outputStatusDetail) outputStatusDetail.textContent = '';
  }

  /* ── Output rendering ──────────────────────────────────────────────────── */

  /** Build a single output-chunk <div> element with toggle listeners attached. */
  function _buildChunkEl(chunk) {
    const el = document.createElement('div');
    const isComplete     = !!chunk.is_complete;
    const shownOriginal  = chunk.mode === 'original';
    const shownBestGated = isComplete && !shownOriginal && chunk.reverted_to_original && chunk.best_gated;
    const statusTone     = !isComplete ? 'pending' : (chunk.reverted_to_original ? 'warn' : 'done');

    let displayLabel, bodyText, metricsHtml, tagClass;
    if (!isComplete && shownOriginal) {
      displayLabel = 'Original';
      tagClass     = 'original';
      bodyText     = chunk.original_text || 'Waiting for original chunk text…';
      metricsHtml  = `
        <div class="metric-row"><span>AI Score</span><strong>—</strong></div>
        <div class="metric-row"><span>Semantic</span><strong>—</strong></div>
      `;
    } else if (!isComplete) {
      displayLabel = 'Processing';
      tagClass     = 'processing';
      bodyText     = 'This chunk will render here as soon as processing completes.';
      metricsHtml  = `
        <div class="metric-row"><span>Status</span><strong>Running</strong></div>
        <div class="metric-row"><span>AI Score</span><strong>—</strong></div>
        <div class="metric-row"><span>Semantic</span><strong>—</strong></div>
      `;
    } else if (shownOriginal) {
      displayLabel = 'Original';
      tagClass     = 'original';
      bodyText     = chunk.original_text;
      metricsHtml  = `
        <div class="metric-row">
          <span>AI Score</span>
          <strong class="metric-ai">${fmtPct(chunk.original_ai_score)}</strong>
        </div>
        <div class="metric-row"><span>Semantic</span><strong>—</strong></div>
      `;
    } else if (shownBestGated) {
      const bg = chunk.best_gated;
      displayLabel = 'Best candidate';
      tagClass     = 'optimized';
      bodyText     = bg.text;
      metricsHtml  = `
        <div class="metric-row">
          <span>AI Score</span>
          <strong class="metric-ai">${fmtPct(bg.ai_score)}</strong>
        </div>
        <div class="metric-row">
          <span>Semantic</span>
          <strong class="metric-sem">${fmtPct(bg.sem_score)}</strong>
        </div>
        <div class="metric-row"><span>Style</span><strong>${escHtml(bg.style)}</strong></div>
        <div class="metric-row" style="font-size:10px;color:var(--yellow)">
          <span>Gate failed</span><strong style="color:var(--yellow)">not used</strong>
        </div>
      `;
    } else {
      displayLabel = 'Model output';
      tagClass     = 'optimized';
      bodyText     = chunk.output_text;
      metricsHtml  = `
        <div class="metric-row">
          <span>AI Score</span>
          <strong class="metric-ai">${fmtPct(chunk.final_ai_score)}</strong>
        </div>
        <div class="metric-row">
          <span>Semantic</span>
          <strong class="metric-sem">${fmtPct(chunk.final_sem_score)}</strong>
        </div>
        ${chunk.selected_style ? `<div class="metric-row"><span>Style</span><strong>${escHtml(chunk.selected_style)}</strong></div>` : ''}
      `;
    }

    const outputBtnLabel = !isComplete
      ? 'Processing'
      : (chunk.reverted_to_original && chunk.best_gated ? 'Best candidate' : 'Output');

    el.className = `output-chunk${chunk.reverted_to_original ? ' reverted' : ''}${!isComplete ? ' pending' : ''}`;
    el.dataset.chunkIdx = chunk.chunk_idx;
    el.innerHTML = `
      <div class="output-chunk-head">
        <div class="output-chunk-title">
          <span>Chunk ${chunk.chunk_idx + 1}</span>
          <span class="output-chunk-tag ${tagClass}">${displayLabel}</span>
          <span class="output-chunk-status ${statusTone}">
            ${escHtml(chunk.status_label || 'Edited')}
          </span>
        </div>
        <div class="output-toggle">
          <button data-mode="original" data-idx="${chunk.chunk_idx}" class="${shownOriginal ? 'active' : ''}">Original</button>
          <button data-mode="output" data-idx="${chunk.chunk_idx}" class="${shownOriginal ? '' : 'active'}" ${!isComplete ? 'disabled' : ''}>${outputBtnLabel}</button>
        </div>
      </div>
      <div class="output-chunk-main">
        <div class="output-chunk-body">${escHtml(bodyText)}</div>
        <div class="output-chunk-metrics">${metricsHtml}</div>
      </div>
    `;
    el.querySelectorAll('.output-toggle button').forEach(btn => {
      btn.addEventListener('click', () => {
        toggleOutputChunk(Number(btn.dataset.idx), btn.dataset.mode);
      });
    });
    return el;
  }

  /**
   * Full re-render of the output view from outputChunks.
   * Used for: initial empty state, plain/chunk mode switch, run_done.
   */
  function renderOutputView() {
    if (!outputView) return;
    if (!outputChunks.length) {
      outputView.className = 'output-view empty';
      outputView.textContent = 'Optimized text appears here...';
      return;
    }

    if (outputRenderMode === 'plain') {
      const outputText = assembleOutputText();
      if (!outputText) {
        outputView.className = 'output-view empty';
        outputView.textContent = 'Completed chunks appear here as they finish...';
        return;
      }
      outputView.className = 'output-view plain';
      outputView.textContent = outputText;
      return;
    }

    outputView.className = 'output-view';
    outputView.innerHTML = '';
    outputChunks.forEach(chunk => outputView.appendChild(_buildChunkEl(chunk)));
  }

  function toggleOutputChunk(idx, mode) {
    const chunk = outputChunks.find(item => item.chunk_idx === idx);
    if (!chunk) return;
    chunk.mode = mode === 'original' ? 'original' : 'output';
    renderOutputView();
  }

  function assembleOutputText() {
    const complete = outputChunks.filter(c => c.is_complete);
    if (!complete.length) return '';

    // Group by para_idx to restore original paragraph boundaries.
    // Chunks from the same paragraph are joined with a space;
    // different paragraphs are separated with a blank line.
    const paraGroups = {};
    complete.forEach(chunk => {
      const pidx = chunk.para_idx ?? chunk.chunk_idx;
      let text;
      if (chunk.mode === 'original') {
        text = chunk.original_text;
      } else if (chunk.reverted_to_original && chunk.best_gated) {
        text = chunk.best_gated.text;
      } else {
        text = chunk.output_text;
      }
      if (!paraGroups[pidx]) paraGroups[pidx] = [];
      paraGroups[pidx].push(text);
    });

    return Object.keys(paraGroups)
      .map(Number)
      .sort((a, b) => a - b)
      .map(pidx => paraGroups[pidx].join(' '))
      .join('\n\n')
      .trim();
  }

  function syncOutputViewButtons() {
    if (viewChunkBtn) viewChunkBtn.classList.toggle('active', outputRenderMode === 'chunk');
    if (viewPlainBtn) viewPlainBtn.classList.toggle('active', outputRenderMode === 'plain');
  }

  /* ── UI helpers ────────────────────────────────────────────────────────── */

  function setRunning(val) {
    running = val;
    runBtn.disabled    = val;
    runBtn.textContent = val ? 'Optimizing…' : 'Optimize';
    const ob  = $('overall-bar');
    const opr = ob && ob.parentElement;
    if (ob) {
      ob.classList.toggle('running', val);
      if (val) {
        ob.style.width = '0%';
        overallVisualPct = 0;
      }
    }
    if (opr) opr.classList.toggle('running', val);
    if (val) {
      startLiveUiLoop();
      wsDot.className     = 'ws-dot running';
      wsLabel.textContent = 'Running…';
    } else {
      stopLiveUiLoop();
      const connected = ws && ws.readyState === WebSocket.OPEN;
      wsDot.className     = connected ? 'ws-dot connected' : 'ws-dot';
      wsLabel.textContent = connected ? 'Connected'        : 'Disconnected';
    }
  }

  function fmtPct(v) { return (v * 100).toFixed(1) + '%'; }

  function timeStamp() {
    return new Date().toLocaleTimeString([], {
      hour12:  false,
      hour:    '2-digit',
      minute:  '2-digit',
      second:  '2-digit',
    });
  }

  function escHtml(s) {
    return String(s)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  /* ── Boot ──────────────────────────────────────────────────────────────── */
  window.__optimizerDebug = {
    dispatchTestMessage(message) {
      dispatch(message);
    },
    getOutputChunks() {
      return JSON.parse(JSON.stringify(outputChunks));
    },
    getOutputMode() {
      return outputRenderMode;
    },
    getOutputHtml() {
      return outputView ? outputView.innerHTML : '';
    },
  };

  connect();

})();
