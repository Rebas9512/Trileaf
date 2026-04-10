/**
 * Trileaf — Dashboard WebSocket client (V2 pipeline)
 *
 * Message types:
 *   stage_start        { stage, name }
 *   stage1_done        { ai_score, violation_count, summary, top_issues, model_useful, doc_mean, doc_std }
 *   sentence_tagged    { idx, text, severity, rule_severity, ai_score, ai_z_score, flags }
 *   stage2_done        { total_sentences, severity_distribution, model_useful }
 *   stage3_progress    { segment_idx, total_segments, status }
 *   stage3_done        { ai_score, violation_count, violation_delta, sem_score }
 *   stage4_sentence    { idx, action, sem_score, detail }
 *   stage4_done        { rewritten, deleted, unfixed, ai_score }
 *   stage5_done        { ai_score, sem_score, techniques_used }
 *   run_done_v2        { run_id, output, original_ai_score, final_ai_score, final_sem_score, unfixed_sentences }
 *   error              { message }
 */

(function () {
  'use strict';

  const $ = id => document.getElementById(id);

  /* ── DOM refs ──────────────────────────────────────────────────────── */
  const inputText       = $('input-text');
  const runBtn          = $('run-btn');
  const modeShortBtn    = $('mode-short-btn');
  const modeLongBtn     = $('mode-long-btn');
  const stagesContainer = $('stages-container');
  const analysisText    = $('analysis-text');
  const outputText      = $('output-text');
  const metricsRow      = $('metrics-row');
  const metricAi        = $('metric-ai');
  const metricSem       = $('metric-sem');
  const metricViol      = $('metric-viol');
  const metricS4        = $('metric-s4');
  const copyBtn         = $('copy-btn');
  const wsDot           = $('ws-dot');
  const wsLabel         = $('ws-label');
  const deviceLabel     = $('device-label');
  const rewriteLabel    = $('rewrite-label');

  /* ── State ─────────────────────────────────────────────────────────── */
  let ws = null;
  let running = false;
  let chunkMode = 'short';

  const STAGE_NAMES = {
    1: 'Global Scoring',
    2: 'Sentence Tagging',
    3: 'Standardized Rewrite',
    4: 'Stubborn Sentences',
    5: 'Human Touch',
    6: 'Formality Calibration',
  };

  let stages = {};        // stage_num → { status, metrics_html, barWidth }
  let sentences = [];     // [ { idx, text, severity, flags, ai_score, ai_z_score } ]
  let unfixedSet = new Set();
  let unfixedTexts = [];   // actual sentence texts for inline highlighting
  let finalOutput = '';
  let s4Stats = { rewritten: 0, deleted: 0, unfixed: 0 };
  let globalAiScore = null;   // Stage 1 overall AI score
  let globalViolCount = 0;
  let modelUseful = true;

  /* ── WebSocket ─────────────────────────────────────────────────────── */
  function connectWS() {
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    ws = new WebSocket(`${proto}://${location.host}/ws/optimizer`);
    ws.onopen = () => { wsDot.className = 'ws-dot connected'; wsLabel.textContent = 'Connected'; };
    ws.onclose = () => { wsDot.className = 'ws-dot'; wsLabel.textContent = 'Disconnected'; setTimeout(connectWS, 2000); };
    ws.onmessage = (ev) => {
      try { const msg = JSON.parse(ev.data); dispatch(msg.type, msg.data); }
      catch (e) { /* ignore */ }
    };
  }

  function dispatch(type, data) {
    switch (type) {
      case 'stage_start':       onStageStart(data); break;
      case 'stage1_done':       onStage1Done(data); break;
      case 'sentence_tagged':   onSentenceTagged(data); break;
      case 'stage2_done':       onStage2Done(data); break;
      case 'stage3_progress':   onStage3Progress(data); break;
      case 'stage3_done':       onStage3Done(data); break;
      case 'stage4_sentence':   onStage4Sentence(data); break;
      case 'stage4_done':       onStage4Done(data); break;
      case 'stage5_done':       onStage5Done(data); break;
      case 'stage6_skipped':    onStage6Skipped(data); break;
      case 'stage6_done':       onStage6Done(data); break;
      case 'run_done_v2':       onRunDoneV2(data); break;
      case 'error':             onError(data); break;
    }
  }

  /* ── Stage card rendering ──────────────────────────────────────────── */
  let liveTimer = null;

  function initStages() {
    stages = {};
    stagesContainer.innerHTML = '';
    stopLiveTicker();
  }

  /** Create or update a single stage card DOM node (avoids full innerHTML rebuild). */
  function ensureStageCard(n) {
    const s = stages[n];
    if (!s) return;
    let card = $(`stage-card-${n}`);
    if (!card) {
      // First time: create the card
      card = document.createElement('div');
      card.id = `stage-card-${n}`;
      card.className = 'stage-card';
      card.innerHTML = `
        <div class="stage-header">
          <div class="stage-dot" id="stage-dot-${n}"></div>
          <div class="stage-name" id="stage-name-${n}">Stage ${n} · ${STAGE_NAMES[n]}</div>
        </div>
        <div class="stage-bar"><div class="stage-bar-fill" id="stage-bar-${n}"></div></div>
        <div class="stage-metrics" id="stage-metrics-${n}"></div>`;
      stagesContainer.appendChild(card);
    }
    // Update classes and content
    const st = s.status;
    const isSkipped = st === 'skipped';
    card.className = `stage-card${st === 'active' ? ' active' : ''}`;
    if (isSkipped) card.style.opacity = '0.45';
    else card.style.opacity = '';

    const dot = $(`stage-dot-${n}`);
    dot.className = `stage-dot ${st}`;
    dot.textContent = st === 'done' ? '✓' : isSkipped ? '—' : '';

    const name = $(`stage-name-${n}`);
    name.className = `stage-name ${isSkipped ? 'waiting' : st}`;

    const bar = $(`stage-bar-${n}`);
    bar.className = `stage-bar-fill ${isSkipped ? 'waiting' : st}`;
    if (st === 'done') bar.style.width = '100%';
    if (isSkipped) bar.style.width = '0%';

    const met = $(`stage-metrics-${n}`);
    met.className = `stage-metrics`;
    met.innerHTML = s.metrics_html || '';
  }

  /** Update all existing stage cards. */
  function renderStages() {
    for (let i = 1; i <= 6; i++) {
      if (stages[i]) ensureStageCard(i);
    }
  }

  /* ── Live progress ticker (V1-inspired exponential easing) ─────────── */

  function startLiveTicker() {
    if (liveTimer) return;
    liveTimer = setInterval(tickLive, 160);
  }

  function stopLiveTicker() {
    if (liveTimer) { clearInterval(liveTimer); liveTimer = null; }
  }

  function tickLive() {
    for (let i = 1; i <= 5; i++) {
      const s = stages[i];
      if (!s || s.status !== 'active') continue;

      // Exponential ease: progress creeps toward ceiling but never reaches it
      const elapsed = (Date.now() - (s.startedAt || Date.now())) / 1000;
      const ease = 1 - Math.exp(-elapsed / 6.0);  // τ=6s, slow creep

      // Floor is set by real events; ceiling depends on stage type
      const floor = s.progressFloor || 5;
      const ceiling = s.progressCeiling || 92;
      const simulated = floor + (ceiling - floor) * ease;
      const pct = Math.min(ceiling, Math.max(floor, simulated));

      const bar = $(`stage-bar-${i}`);
      if (bar) bar.style.width = `${pct.toFixed(1)}%`;
    }
  }

  /* ── Event handlers ────────────────────────────────────────────────── */
  function onStageStart(d) {
    const n = d.stage;
    // Complete all previous stages
    for (let i = 1; i < n; i++) {
      if (stages[i] && stages[i].status !== 'done') {
        stages[i].status = 'done';
        stages[i].progressFloor = 100;
      }
    }
    // Create the new active stage
    stages[n] = stages[n] || {};
    stages[n].status = 'active';
    stages[n].startedAt = Date.now();
    stages[n].progressFloor = 5;
    stages[n].progressCeiling = 92;
    stages[n].metrics_html = 'Processing...';
    renderStages();
    startLiveTicker();
  }

  function onStage1Done(d) {
    globalAiScore = d.ai_score;
    globalViolCount = d.violation_count;
    modelUseful = d.model_useful !== false;
    stages[1].status = 'done';
    stages[1].progressFloor = 100;
    const genreTag = d.genre ? `<span style="color:var(--accent)">${d.genre}</span>  ·  ` : '';
    stages[1].metrics_html = `${genreTag}AI: ${d.ai_score.toFixed(2)}  ·  Violations: ${d.violation_count}`;
    if (!d.model_useful) {
      stages[1].metrics_html += '  ·  <span style="color:#fbbf24">Low disc.</span>';
    }
    renderStages();
    renderAnalysisHeader();
  }

  function onSentenceTagged(d) {
    sentences.push({
      idx: d.idx,
      text: d.text,
      severity: d.severity,
      flags: d.flags || [],
      ai_score: d.ai_score != null ? d.ai_score : null,
      ai_z_score: d.ai_z_score != null ? d.ai_z_score : null,
    });
    // Render incrementally as each sentence arrives
    renderAnalysis();
  }

  function onStage2Done(d) {
    stages[2].status = 'done';
    stages[2].progressFloor = 100;
    const dist = d.severity_distribution || {};
    const parts = [];
    if (dist.critical) parts.push(`C:${dist.critical}`);
    if (dist.high) parts.push(`H:${dist.high}`);
    if (dist.medium) parts.push(`M:${dist.medium}`);
    if (dist.low) parts.push(`L:${dist.low}`);
    if (dist.clean) parts.push(`Clean:${dist.clean}`);
    stages[2].metrics_html = parts.join('  ') || 'Done';
    renderStages();
    renderAnalysis();
  }

  function onStage3Progress(d) {
    // Raise the progress floor based on real segment completion
    const pct = Math.round(((d.segment_idx + 1) / d.total_segments) * 90);
    if (stages[3]) {
      stages[3].progressFloor = Math.max(stages[3].progressFloor || 0, pct);
      stages[3].startedAt = stages[3].startedAt || Date.now(); // reset ease curve
    }
    stages[3].metrics_html = `Rewriting segment ${d.segment_idx + 1}/${d.total_segments}...`;
    renderStages();
  }

  function onStage3Done(d) {
    stages[3].status = 'done';
    stages[3].progressFloor = 100;
    stages[3].metrics_html = `AI: ${d.ai_score.toFixed(2)}  ·  Δviol: ${d.violation_delta}  ·  Sem: ${d.sem_score.toFixed(2)}`;
    renderStages();
  }

  function onStage4Sentence(d) {
    if (d.action === 'unfixed_risky') unfixedSet.add(d.idx);
  }

  function onStage4Done(d) {
    stages[4].status = 'done';
    stages[4].progressFloor = 100;
    s4Stats = { rewritten: d.rewritten || 0, deleted: d.deleted || 0, unfixed: d.unfixed || 0 };
    const parts = [];
    if (s4Stats.rewritten) parts.push(`Rewrite ${s4Stats.rewritten}`);
    if (s4Stats.deleted) parts.push(`Del ${s4Stats.deleted}`);
    if (s4Stats.unfixed) parts.push(`⚠ ${s4Stats.unfixed}`);
    stages[4].metrics_html = parts.join(' · ') || 'No stubborn sentences';
    renderStages();
  }

  function onStage5Done(d) {
    stages[5].status = 'done';
    stages[5].progressFloor = 100;
    stages[5].metrics_html = `AI: ${d.ai_score.toFixed(2)}  ·  Sem: ${d.sem_score.toFixed(2)}`;
    renderStages();
  }

  function onStage6Skipped(d) {
    // Show a "skipped" card for Stage 6
    stages[6] = {
      status: 'skipped',
      progressFloor: 0,
      metrics_html: `<span style="color:var(--muted)">Skipped (${d.genre} genre)</span>`,
    };
    ensureStageCard(6);
  }

  function onStage6Done(d) {
    stages[6].status = 'done';
    stages[6].progressFloor = 100;
    const accepted = d.accepted ? '<span style="color:var(--green)">✓ accepted</span>' : '<span style="color:var(--yellow)">✗ reverted</span>';
    stages[6].metrics_html = `AI: ${d.ai_score.toFixed(2)}  ·  Sem: ${d.sem_score.toFixed(2)}  ·  ${accepted}`;
    renderStages();
  }

  function onRunDoneV2(d) {
    running = false;
    runBtn.disabled = false;
    runBtn.textContent = 'Optimize';
    stopLiveTicker();
    finalOutput = d.output || '';

    if (d.unfixed_sentences) d.unfixed_sentences.forEach(i => unfixedSet.add(i));
    unfixedTexts = d.unfixed_texts || [];

    // Metrics cards
    metricsRow.style.display = 'flex';
    const origAi = d.original_ai_score;
    const finalAi = d.final_ai_score;
    const pct = origAi > 0 ? Math.round((1 - finalAi / origAi) * 100) : 0;
    metricAi.innerHTML = `<span class="from">${origAi.toFixed(2)}</span><span class="arrow">→</span><span class="to">${finalAi.toFixed(2)}</span><span class="pct">↓${pct}%</span>`;
    metricSem.innerHTML = `<span class="single">${d.final_sem_score.toFixed(2)}</span>`;
    metricViol.innerHTML = `<span class="from">${globalViolCount}</span><span class="arrow">→</span><span class="to">0</span>`;

    const s4p = [];
    if (s4Stats.rewritten) s4p.push(`Rewrite ${s4Stats.rewritten}`);
    if (s4Stats.deleted) s4p.push(`Del ${s4Stats.deleted}`);
    if (s4Stats.unfixed) s4p.push(`⚠ ${s4Stats.unfixed}`);
    metricS4.innerHTML = `<span class="s4-detail">${s4p.join(' · ') || '—'}</span>`;

    renderOutput();
  }

  function onError(d) {
    running = false;
    runBtn.disabled = false;
    runBtn.textContent = 'Optimize';
    stopLiveTicker();
    outputText.innerHTML = `<span style="color:var(--red)">Error: ${d.message || 'Unknown error'}</span>`;
  }

  /* ── Rendering ─────────────────────────────────────────────────────── */

  function renderAnalysisHeader() {
    // Prepend overall score badge if we have it
    if (globalAiScore == null) return;
    const badge = document.querySelector('.analysis-score-badge');
    if (badge) badge.remove();

    const header = document.querySelector('.analysis-pane .pane-header .section-label');
    if (!header) return;
    const span = document.createElement('span');
    span.className = 'analysis-score-badge';
    const color = globalAiScore > 0.6 ? 'var(--red)' : globalAiScore > 0.4 ? 'var(--yellow)' : 'var(--green)';
    span.innerHTML = ` <span style="font-family:'Geist Mono',monospace;font-weight:600;color:${color};margin-left:8px">Overall AI: ${globalAiScore.toFixed(2)}</span>`;
    header.appendChild(span);
  }

  function renderAnalysis() {
    if (!sentences.length) return;

    let html = '';
    for (let i = 0; i < sentences.length; i++) {
      const s = sentences[i];
      if (i > 0) html += ' ';

      const hlClass = getHighlightClass(s.severity);

      // Build rich tooltip: flags + AI score + z-score
      let tipParts = [];
      if (s.flags.length) tipParts.push(s.flags.join(', '));
      if (s.ai_score != null) tipParts.push(`AI: ${s.ai_score.toFixed(3)}`);
      if (modelUseful && s.ai_z_score != null) {
        tipParts.push(`z: ${s.ai_z_score.toFixed(2)}`);
      } else if (!modelUseful && s.ai_score != null) {
        tipParts.push('(low model discrimination)');
      }
      const tooltip = tipParts.length ? ` title="${escAttr(tipParts.join('  |  '))}"` : '';

      if (hlClass) {
        html += `<span class="${hlClass}"${tooltip}>${escHtml(s.text)}</span>`;
      } else {
        html += escHtml(s.text);
      }
    }
    analysisText.innerHTML = html;
  }

  function renderOutput() {
    if (!finalOutput) return;

    if (unfixedTexts.length > 0) {
      // Highlight unfixed sentences within the full output text
      let raw = finalOutput;
      let segments = [];  // [{text, unfixed}]
      // Sort by length descending to match longer sentences first
      const sorted = [...unfixedTexts].sort((a, b) => b.length - a.length);

      // Mark positions of unfixed sentences
      let marks = [];  // [{start, end}]
      for (const ut of sorted) {
        const idx = raw.indexOf(ut);
        if (idx >= 0) {
          marks.push({ start: idx, end: idx + ut.length });
        }
      }
      marks.sort((a, b) => a.start - b.start);

      // Build segments
      let cursor = 0;
      for (const m of marks) {
        if (m.start > cursor) {
          segments.push({ text: raw.slice(cursor, m.start), unfixed: false });
        }
        segments.push({ text: raw.slice(m.start, m.end), unfixed: true });
        cursor = m.end;
      }
      if (cursor < raw.length) {
        segments.push({ text: raw.slice(cursor), unfixed: false });
      }

      let html = '';
      for (const seg of segments) {
        const escaped = escHtml(seg.text).replace(/\n\n/g, '<br><br>').replace(/\n/g, '<br>');
        if (seg.unfixed) {
          html += `<span class="hl-unfixed">${escaped}<span class="unfixed-tag">⚠ review</span></span>`;
        } else {
          html += escaped;
        }
      }
      outputText.innerHTML = html;
    } else {
      outputText.innerHTML = escHtml(finalOutput).replace(/\n\n/g, '<br><br>').replace(/\n/g, '<br>');
    }
  }

  function getHighlightClass(severity) {
    switch (severity) {
      case 'critical': return 'hl-critical';
      case 'high':     return 'hl-high';
      case 'medium':   return 'hl-medium';
      default:         return '';
    }
  }

  function escHtml(s) {
    const d = document.createElement('div');
    d.textContent = s;
    return d.innerHTML;
  }

  function escAttr(s) {
    return s.replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }

  /* ── Actions ───────────────────────────────────────────────────────── */
  function startRun() {
    if (running) return;
    const text = inputText.value.trim();
    if (!text) return;

    running = true;
    runBtn.disabled = true;
    runBtn.textContent = 'Running...';

    // Reset
    sentences = [];
    unfixedSet = new Set();
    unfixedTexts = [];
    finalOutput = '';
    s4Stats = { rewritten: 0, deleted: 0, unfixed: 0 };
    globalAiScore = null;
    globalViolCount = 0;
    modelUseful = true;
    metricsRow.style.display = 'none';
    analysisText.innerHTML = '<span class="placeholder-msg">Analyzing...</span>';
    outputText.innerHTML = '<span class="placeholder-msg">Waiting for pipeline to complete...</span>';
    // Remove old score badge
    const oldBadge = document.querySelector('.analysis-score-badge');
    if (oldBadge) oldBadge.remove();

    initStages();

    fetch('/api/optimize', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, chunk_mode: chunkMode }),
    }).catch(() => {
      running = false;
      runBtn.disabled = false;
      runBtn.textContent = 'Optimize';
    });
  }

  /* ── Mode toggle ───────────────────────────────────────────────────── */
  modeShortBtn.addEventListener('click', () => {
    chunkMode = 'short';
    modeShortBtn.classList.add('active');
    modeLongBtn.classList.remove('active');
  });
  modeLongBtn.addEventListener('click', () => {
    chunkMode = 'long';
    modeLongBtn.classList.add('active');
    modeShortBtn.classList.remove('active');
  });

  runBtn.addEventListener('click', startRun);

  copyBtn.addEventListener('click', () => {
    if (finalOutput) {
      navigator.clipboard.writeText(finalOutput);
      copyBtn.textContent = 'Copied!';
      setTimeout(() => { copyBtn.textContent = 'Copy'; }, 1500);
    }
  });

  /* ── Health info ───────────────────────────────────────────────────── */
  fetch('/api/health').then(r => r.json()).then(d => {
    deviceLabel.textContent = d.device || '—';
    rewriteLabel.textContent = d.rewrite_model || '—';
  }).catch(() => {});

  /* ── Boot ──────────────────────────────────────────────────────────── */
  initStages();
  connectWS();

})();
