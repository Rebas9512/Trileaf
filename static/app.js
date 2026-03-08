/**
 * LLM Writing Optimizer — Dashboard WebSocket client
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
  const wAiSlider          = $('w-ai-slider');
  const wSemSlider         = $('w-sem-slider');
  const wRiskSlider        = $('w-risk-slider');
  const wAiVal             = $('w-ai-val');
  const wSemVal            = $('w-sem-val');
  const wRiskVal           = $('w-risk-val');
  const weightSumBadge     = $('weight-sum-badge');
  const weightSumHint      = $('weight-sum-hint');
  const runBtn             = $('run-btn');
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
  const qwenLabel          = $('qwen-label');

  /* ── State ─────────────────────────────────────────────────────────────── */
  let ws          = null;
  let running     = false;
  const cards     = {};   // chunk_idx → { el, logCount }
  let totalChunks = 0;
  let doneChunks  = 0;
  let outputChunks = [];
  let outputRenderMode = 'chunk';

  /* ── Weight sliders — sum validation ──────────────────────────────────── */
  const weightSliders = [wAiSlider, wSemSlider, wRiskSlider];
  const weightRawVals = [wAiVal,    wSemVal,    wRiskVal   ];

  function weightSum() {
    return weightSliders.reduce((acc, s) => acc + parseFloat(s.value), 0);
  }

  function updateWeightDisplay() {
    const raws = weightSliders.map(s => parseFloat(s.value));
    raws.forEach((r, i) => { weightRawVals[i].textContent = r.toFixed(2); });

    const total = raws.reduce((a, b) => a + b, 0);
    const ok    = Math.abs(total - 1.0) < 0.005;
    if (weightSumBadge) {
      weightSumBadge.textContent = total.toFixed(2);
      weightSumBadge.className   = `weight-sum-badge ${ok ? 'ok' : 'err'}`;
    }
    if (weightSumHint) {
      weightSumHint.textContent = ok ? '' : `(需要 = 1.00，差 ${(total - 1.0).toFixed(2)})`;
    }
    // Disable run button if weights are wrong
    runBtn.disabled = running || !ok;
  }

  weightSliders.forEach(s => s.addEventListener('input', updateWeightDisplay));
  updateWeightDisplay();  // initialise display

  function getWeights() {
    return {
      w_ai:   parseFloat(wAiSlider.value),
      w_sem:  parseFloat(wSemSlider.value),
      w_risk: parseFloat(wRiskSlider.value),
    };
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

  /* ── Fetch health info ─────────────────────────────────────────────────── */
  function fetchHealth() {
    fetch('/api/health')
      .then(r => r.json())
      .then(d => {
        deviceLabel.textContent = d.device || '?';
        qwenLabel.textContent   = d.qwen_backend || '?';
      })
      .catch(() => {});
  }

  /* ── WebSocket ─────────────────────────────────────────────────────────── */
  function connect() {
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    ws = new WebSocket(`${proto}://${location.host}/ws/optimizer`);

    ws.onopen = () => {
      wsDot.className     = 'ws-dot connected';
      wsLabel.textContent = 'Connected';
      fetchHealth();
    };

    ws.onclose = () => {
      wsDot.className     = 'ws-dot';
      wsLabel.textContent = 'Disconnected';
      setTimeout(connect, 3000);
    };

    ws.onerror = () => {
      wsDot.className     = 'ws-dot';
      wsLabel.textContent = 'Error';
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

  function onRunStart(data) {
    Object.keys(cards).forEach(k => delete cards[k]);
    chunkList.innerHTML = '';
    if (progressPH) {
      chunkList.appendChild(progressPH);
      progressPH.style.display = 'none';
    }

    outputChunks = [];
    renderOutputView();
    clearOutputStatus();
    outputRenderMode = 'chunk';
    syncOutputViewButtons();

    const ob = $('overall-bar');
    if (ob) ob.style.width = '0%';

    totalChunks = data.total_chunks;
    doneChunks  = 0;
    runSummary.textContent = `0 / ${totalChunks} chunks done`;

    setRunning(true);
  }

  function onChunkStart(data) {
    const card = buildCard(data.chunk_idx, data.total, data.text);
    cards[data.chunk_idx] = card;
    chunkList.appendChild(card.el);
    card.el.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
    setStage(data.chunk_idx, 'rewrite', 'active');
    setPhase(data.chunk_idx, 'Ensemble', 'Generating 3 rewrite candidates');
    setStatus(data.chunk_idx, 'Rewriting…');
  }

  function onEnsembleCandidates(data) {
    // data.candidates: [{style, ai_score, sem_score, ppl, ...}]
    const scores = (data.candidates || [])
      .map(c => `${c.style[0].toUpperCase()}: ${fmtPct(c.ai_score)}`)
      .join(' · ');
    const best = (data.candidates || []).reduce(
      (a, b) => (!a || b.ai_score < a.ai_score) ? b : a, null
    );
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
        const conn = getConn(data.chunk_idx, 'rewrite');
        if (conn) conn.classList.add('done');
      }
    }
    if (data.stage === 'batch_score') {
      setStage(data.chunk_idx, 'batch_score', data.state);
      if (data.state === 'done' || data.state === 'warn') {
        const conn = getConn(data.chunk_idx, 'batch_score');
        if (conn) conn.classList.add('done');
      }
    }
    if (data.stage === 'pareto') {
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

    const card = cards[data.chunk_idx];
    if (card) {
      card.el.classList.remove('active-card');
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

    const ob = $('overall-bar');
    if (ob) ob.style.width = (doneChunks / totalChunks * 100).toFixed(1) + '%';
  }

  function onFinalScoring() {
    runSummary.textContent = 'Computing final scores…';
  }

  function onRunDone(data) {
    // ── FIX: new field is final_text, not output_text ──
    outputChunks = (data.chunks || []).map(chunk => ({
      chunk_idx:            chunk.chunk_idx,
      original_text:        chunk.original_text    || '',
      output_text:          chunk.final_text        || '',
      reverted_to_original: !!chunk.reverted_to_original,
      best_gated:           chunk.best_candidate   || null,
      status_label:         chunk.status_label      || (chunk.reverted_to_original ? 'Reverted to original' : 'Edited'),
      original_ai_score:    chunk.original_ai_score || 0,
      final_ai_score:       chunk.final_ai_score    || 0,
      final_sem_score:      chunk.final_sem_score   || 0,
      selected_style:       chunk.selected_style    || '',
      mode: chunk.reverted_to_original ? 'original' : 'output',
    }));
    renderOutputView();

    const revertedCount = (data.chunks || []).filter(c => c.reverted_to_original).length;

    // AI score before → after with reduction delta
    const origAi   = data.original_ai_score || 0;
    const finalAi  = data.final_ai_score    || 0;
    const delta    = (origAi - finalAi) * 100;   // positive = improvement
    const deltaStr = delta >= 0
      ? `<span style="color:var(--green)">↓${delta.toFixed(1)}pp</span>`
      : `<span style="color:var(--red)">↑${Math.abs(delta).toFixed(1)}pp</span>`;
    // labelHtml — passes raw HTML so numbers can be styled
    // Use <span> (not <strong>) to avoid the .output-status strong CSS color override
    const aiCompare = `<span style="color:var(--muted)">AI detection &nbsp;</span>`
      + `<span style="font-weight:700;color:#ffb0b0">${fmtPct(origAi)}</span>`
      + `<span style="color:var(--muted)"> before &nbsp;→&nbsp; </span>`
      + `<span style="font-weight:700;color:var(--green)">${fmtPct(finalAi)}</span>`
      + `<span style="color:var(--muted)"> after &nbsp;</span>`
      + deltaStr;
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

  /* ── Run button ────────────────────────────────────────────────────────── */
  runBtn.addEventListener('click', async () => {
    const text = inputText.value.trim();
    if (!text) { inputText.focus(); return; }

    // Hard check: weights must sum to 1.0
    const total = weightSum();
    if (Math.abs(total - 1.0) >= 0.005) {
      alert(`权重必须相加得 1.00（当前合计 ${total.toFixed(2)}）\n请调整各权重滑块使总和等于 1.00`);
      return;
    }

    setRunning(true);

    try {
      const res = await fetch('/api/optimize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, ...getWeights() }),
      });
      const data = await res.json();
      if (data.error) { alert(data.error); setRunning(false); }
    } catch (e) {
      alert('Failed to start optimization: ' + e.message);
      setRunning(false);
    }
  });

  /* ── Card helpers ──────────────────────────────────────────────────────── */

  function buildCard(idx, total, previewText) {
    const el = document.createElement('div');
    el.className = 'chunk-card active-card';
    el.dataset.idx = idx;
    el.innerHTML = `
      <div class="chunk-head">
        <span class="chunk-title">Chunk ${idx + 1} / ${total}</span>
        <span class="chunk-status" data-status="${idx}">Queued…</span>
      </div>
      <div class="chunk-preview">${escHtml(previewText)}</div>
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
    return { el, logCount: 0 };
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
      labelEl.className = `phase-badge${tone ? ' ' + tone : ''}`;
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

    while (list.children.length > 8) list.removeChild(list.lastChild);

    card.logCount = (card.logCount || 0) + 1;
    if (counter) counter.textContent = `${Math.min(card.logCount, 8)} shown`;
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

  function renderOutputView() {
    if (!outputView) return;
    if (!outputChunks.length) {
      outputView.className = 'output-view empty';
      outputView.textContent = 'Optimized text appears here...';
      return;
    }

    if (outputRenderMode === 'plain') {
      outputView.className = 'output-view plain';
      outputView.textContent = assembleOutputText();
      return;
    }

    outputView.className = 'output-view';
    outputView.innerHTML = '';
    outputChunks.forEach(chunk => {
      const el = document.createElement('div');
      const shownOriginal  = chunk.mode === 'original';
      // For reverted chunks: 'output' mode shows best_gated (if available)
      const shownBestGated = !shownOriginal && chunk.reverted_to_original && chunk.best_gated;
      const statusTone     = chunk.status_label === 'Edited' ? 'done' : 'warn';

      let displayLabel, bodyText, metricsHtml, tagClass;
      if (shownOriginal) {
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

      const outputBtnLabel = chunk.reverted_to_original && chunk.best_gated
        ? 'Best candidate' : 'Output';

      el.className = `output-chunk${chunk.reverted_to_original ? ' reverted' : ''}`;
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
            <button data-mode="output"   data-idx="${chunk.chunk_idx}" class="${shownOriginal ? '' : 'active'}">${outputBtnLabel}</button>
          </div>
        </div>
        <div class="output-chunk-main">
          <div class="output-chunk-body">${escHtml(bodyText)}</div>
          <div class="output-chunk-metrics">${metricsHtml}</div>
        </div>
      `;
      outputView.appendChild(el);
    });

    outputView.querySelectorAll('.output-toggle button').forEach(btn => {
      btn.addEventListener('click', () => {
        toggleOutputChunk(Number(btn.dataset.idx), btn.dataset.mode);
      });
    });
  }

  function toggleOutputChunk(idx, mode) {
    const chunk = outputChunks.find(item => item.chunk_idx === idx);
    if (!chunk) return;
    chunk.mode = mode === 'original' ? 'original' : 'output';
    renderOutputView();
  }

  function assembleOutputText() {
    return outputChunks.map(chunk => {
      if (chunk.mode === 'original') return chunk.original_text;
      if (chunk.reverted_to_original && chunk.best_gated) return chunk.best_gated.text;
      return chunk.output_text;
    }).join('\n\n').trim();
  }

  function syncOutputViewButtons() {
    if (viewChunkBtn) viewChunkBtn.classList.toggle('active', outputRenderMode === 'chunk');
    if (viewPlainBtn) viewPlainBtn.classList.toggle('active', outputRenderMode === 'plain');
  }

  /* ── UI helpers ────────────────────────────────────────────────────────── */

  function setRunning(val) {
    running = val;
    const weightsOk = Math.abs(weightSum() - 1.0) < 0.005;
    runBtn.disabled    = val || !weightsOk;
    runBtn.textContent = val ? 'Optimizing…' : 'Optimize';
    if (val) {
      wsDot.className     = 'ws-dot running';
      wsLabel.textContent = 'Running…';
    } else {
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
  connect();

})();
