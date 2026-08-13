"use strict";

/* ============================================================
   Mule-Hunt · investigation console
   Vanilla JS, no framework. Fetches the FastAPI risk API and
   renders a premium, grounded, accessible dashboard.
   ============================================================ */

const $  = (id) => document.getElementById(id);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

const fmt = new Intl.NumberFormat("en-IN");
const fmtIN = (n) => fmt.format(n);
const pct = (n) => `${(n * 100).toFixed(1)}%`;
const money = (n) => "₹" + fmtIN(Math.round(n));

const esc = (s) =>
  String(s).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));

const bandClass = (b) =>
  b === "high" ? "band-high" : b === "medium" ? "band-medium" : "band-low";

const state = {
  selectedAccount: null,
  selectedRing: null,
  top: [],
  rings: {},
  summary: null,
};

/* ---------- fetch helpers ---------- */
async function getJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${r.status} ${url}`);
  return r.json();
}
async function postJSON(url, body) {
  const r = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) {
    const detail = await r.json().catch(() => ({}));
    throw new Error(detail.detail || `${r.status}`);
  }
  return r.json();
}

/* ---------- theme ---------- */
const THEME_KEY = "mule-hunt-theme";
function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  try { localStorage.setItem(THEME_KEY, theme); } catch {}
}
function initTheme() {
  let saved = "light";
  try { saved = localStorage.getItem(THEME_KEY) || "light"; } catch {}
  applyTheme(saved);
}
$("theme-toggle")?.addEventListener("click", () => {
  const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
  applyTheme(next);
});

/* ---------- toast ---------- */
const toastEl = $("toast");
let toastTimer = null;
function toast(msg) {
  toastEl.textContent = msg;
  toastEl.hidden = false;
  requestAnimationFrame(() => toastEl.classList.add("show"));
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => {
    toastEl.classList.remove("show");
    setTimeout(() => (toastEl.hidden = true), 250);
  }, 2200);
}

/* ---------- summary / KPIs / brand ---------- */
async function loadSummary() {
  const s = await getJSON("/api/summary");
  state.summary = s;
  $("kpi-accounts").textContent = fmtIN(s.n_accounts);
  $("kpi-accounts-sub").textContent = `nodes in the deployed graph`;
  $("kpi-tx").textContent = fmtIN(s.n_transactions);
  $("kpi-tx-sub").textContent = `directed transfers (test ring-aware split)`;
  $("kpi-fraud").textContent = fmtIN(s.n_fraud);
  $("kpi-fraud-sub").textContent = `${pct(s.fraud_rate)} of accounts`;
  $("kpi-rings").textContent = fmtIN(s.n_rings);
  $("kpi-rings-sub").textContent = `coordinated fraud cycles`;

  $("brand-model").textContent = `model · ${s.model} (L=${s.num_layers})`;
  $("brand-fraud").textContent = `${pct(s.fraud_rate)} fraud rate`;
  $("brand-version").textContent = `v${s.mule_hunt_version || "?"}`;

  // Populate ring selector
  const sel = $("ring-select");
  sel.innerHTML = "";
  for (const [id, size] of Object.entries(s.ring_sizes || {})) {
    const opt = document.createElement("option");
    opt.value = id;
    opt.textContent = `Ring ${id} · ${size} accounts`;
    sel.appendChild(opt);
  }
  if (sel.options.length) {
    sel.value = Object.keys(s.ring_sizes)[0];
    state.selectedRing = Number(sel.value);
    loadRing(state.selectedRing);
  }
  sel.addEventListener("change", () => {
    state.selectedRing = Number(sel.value);
    loadRing(state.selectedRing);
  });
}

/* ---------- top risk table ---------- */
function renderTopRow(r, selectedId) {
  const band = r.risk_band || bandOf(r.risk_score);
  const ringLabel = r.ring_id >= 0
    ? `<span class="tag tag-ring">ring ${r.ring_id}</span>`
    : `<span class="tag" style="color:var(--muted)">—</span>`;
  const label = r.true_label
    ? `<span class="tag tag-fraud">fraud</span>`
    : `<span class="tag" style="color:var(--muted)">—</span>`;
  const cls = selectedId === r.account_id ? "selected" : "";
  return `
    <tr data-aid="${esc(r.account_id)}" class="${cls}">
      <td class="rank">${r.rank}</td>
      <td class="acct">${esc(r.account_id)}</td>
      <td class="risk">
        <div class="score-bar">
          <div class="score-bar-track"><div class="score-bar-fill" style="width:${(r.risk_score*100).toFixed(0)}%"></div></div>
          <span class="score-bar-val">${r.risk_score.toFixed(3)}</span>
        </div>
      </td>
      <td><span class="band ${bandClass(band)}">${band}</span></td>
      <td>${ringLabel} ${label}</td>
      <td class="deg">${r.degree}</td>
    </tr>`;
}
function bandOf(score) {
  return score >= 0.7 ? "high" : score >= 0.4 ? "medium" : "low";
}
async function loadTop() {
  const rows = await getJSON("/api/top?k=50");
  state.top = rows;
  const tbody = $("top-tbody");
  tbody.innerHTML = rows.map((r) => renderTopRow(r, state.selectedAccount)).join("");
  tbody.addEventListener("click", onTopClick);
  updateSuggestedChip(rows);
}
function updateSuggestedChip(rows) {
  const top = (rows || []).find((r) => r.account_id != null);
  if (!top) return;
  const chip = document.querySelector('.chip-suggest[data-q="why is acc_7 risky?"]');
  if (!chip) return;
  chip.dataset.q = `why is ${top.account_id} risky?`;
  chip.innerHTML = `Why is <code>${esc(top.account_id)}</code> risky?`;
}
function onTopClick(ev) {
  const tr = ev.target.closest("tr[data-aid]");
  if (!tr) return;
  const aid = tr.dataset.aid;
  selectAccount(aid);
}
function selectAccount(aid) {
  state.selectedAccount = aid;
  $$("#top-tbody tr").forEach((r) => r.classList.toggle("selected", r.dataset.aid === aid));
  loadCanvas(aid);
}

/* ---------- ring explorer ---------- */
async function loadRing(ringId) {
  const side = $("ring-side");
  side.innerHTML = `<div class="skeleton skeleton-block"></div>`;
  try {
    const r = await getJSON(`/api/ring/${ringId}`);
    drawRing(r);
    side.innerHTML = renderRingSide(r);
    $$(".ring-side-row").forEach((el) =>
      el.addEventListener("click", () => selectAccount(el.dataset.aid))
    );
    $$("#ring-svg circle.node").forEach((el) =>
      el.addEventListener("click", () => selectAccount(el.dataset.aid))
    );
  } catch (e) {
    side.innerHTML = `<div class="canvas-empty"><p>Failed to load ring ${ringId}.</p></div>`;
  }
}
function renderRingSide(r) {
  const risky = (r.transactions || []).filter((t) => t.risk_score != null)
    .sort((a, b) => (b.risk_score || 0) - (a.risk_score || 0))
    .slice(0, 5);
  const members = r.nodes
    .slice()
    .sort((a, b) => b.risk_score - a.risk_score)
    .map((n) => `
      <div class="ring-side-row" data-aid="${esc(n.account_id)}" title="${esc(n.account_id)}">
        <div>
          <div class="acc">${esc(n.account_id)}</div>
          <div class="meta">${n.true_label ? "fraud label" : "normal"} · ext ${n.external_connections}</div>
        </div>
        <div style="text-align:right">
          <div class="num">${n.risk_score.toFixed(3)}</div>
          <div><span class="band ${bandClass(bandOf(n.risk_score))}">${bandOf(n.risk_score)}</span></div>
        </div>
      </div>`).join("");
  const tx = risky.length ? `
    <div class="ring-side-h">Top suspicious transactions</div>
    ${risky.map((t) => `
      <div class="ring-side-row">
        <div>
          <div class="acc">a${t.src} → a${t.dst}</div>
          <div class="meta">${money(t.amount)}</div>
        </div>
        <div style="text-align:right">
          <div class="num">${(t.risk_score || 0).toFixed(3)}</div>
          <div class="meta">tx risk</div>
        </div>
      </div>`).join("")}
  ` : "";
  return `<div class="ring-side-h">Ring ${r.ring_id} · ${r.size} members</div>${members}${tx}`;
}
function drawRing(r) {
  const svg = $("ring-svg");
  const cx = 210, cy = 210, radius = 150;
  const n = r.nodes.length;
  const pos = {};
  r.nodes.forEach((node, i) => {
    const angle = (2 * Math.PI * i) / n - Math.PI / 2;
    pos[node.index] = { x: cx + radius * Math.cos(angle), y: cy + radius * Math.sin(angle) };
  });

  const txnRisk = {};
  for (const t of r.transactions || []) {
    txnRisk[`${t.src},${t.dst}`] = t;
  }
  const W = 420, H = 420;
  let inner = `<circle class="ring-bg" cx="${cx}" cy="${cy}" r="${radius+8}" />`;
  for (const [a, b] of r.edges) {
    const t = txnRisk[`${a},${b}`] || { risk_score: 0 };
    const cls = t.risk_score >= 0.5 ? "hot" : t.risk_score >= 0.1 ? "medium" : "";
    inner += `<line class="edge ${cls}" x1="${pos[a].x.toFixed(1)}" y1="${pos[a].y.toFixed(1)}" x2="${pos[b].x.toFixed(1)}" y2="${pos[b].y.toFixed(1)}">
      <title>a${a} → a${b} · risk ${(t.risk_score ?? 0).toFixed(3)} · ${money(t.amount ?? 0)}</title>
    </line>`;
  }
  r.nodes.forEach((node, i) => {
    const p = pos[node.index];
    const cls = node.true_label ? "node node-fraud" : "node node-clean";
    const r2 = node.risk_score >= 0.7 ? 13 : node.risk_score >= 0.4 ? 10 : 7;
    const selected = state.selectedAccount === node.account_id ? "selected" : "";
    const animDelay = `${i * 30}ms`;
    inner += `<circle class="${cls} ${selected}" data-aid="${esc(node.account_id)}" cx="${p.x.toFixed(1)}" cy="${p.y.toFixed(1)}" r="${r2}" style="animation: bubble-in 280ms var(--ease-out) ${animDelay} both">
      <title>${esc(node.account_id)} · ${node.risk_score.toFixed(3)}</title>
    </circle>`;
    inner += `<text x="${p.x.toFixed(1)}" y="${(p.y + r2 + 12).toFixed(1)}" text-anchor="middle">${esc(node.account_id.replace("acc_", "a"))}</text>`;
  });
  svg.innerHTML = inner;
}

/* ---------- canvas (account deep dive) ---------- */
async function loadCanvas(aid) {
  $("canvas-actions").hidden = false;
  const body = $("canvas-body");
  body.innerHTML = `<div class="skeleton skeleton-block"></div>`;
  try {
    const a = await getJSON(`/api/account/${encodeURIComponent(aid)}`);
    body.innerHTML = renderCanvas(a);
    $("btn-explain").onclick = () => explain(a.account_id);
    $("btn-counterfactual").onclick = () => showCounterfactual(a.account_id);
    $("btn-case").onclick = () => openCase(a.account_id);
    $$(".nbr-row").forEach((el) =>
      el.addEventListener("click", () => selectAccount(el.dataset.aid))
    );
  } catch (e) {
    body.innerHTML = `<div class="canvas-empty"><p>Failed to load ${esc(aid)}.</p></div>`;
  }
}
function renderCanvas(a) {
  const band = a.risk_band || bandOf(a.risk_score);
  const pctScore = Math.round(a.risk_score * 100);
  const total = state.summary ? state.summary.n_accounts : null;
  const rankTxt = total ? `${fmtIN(a.rank)} / ${fmtIN(total)}` : fmtIN(a.rank);
  const ring = a.ring_id >= 0 ? `<span class="chip tag-ring">ring ${a.ring_id}</span>` : "";
  const fraud = a.true_label ? `<span class="chip tag-fraud">fraud label</span>` : "";

  const neighbors = a.neighbors.slice()
    .sort((x, y) => y.risk_score - x.risk_score)
    .slice(0, 8)
    .map((n) => `
      <div class="nbr-row" data-aid="${esc(n.account_id)}">
        <div class="acc">${esc(n.account_id)}</div>
        <span class="band ${bandClass(bandOf(n.risk_score))}">${bandOf(n.risk_score)}</span>
        <span class="risk">${n.risk_score.toFixed(3)}</span>
      </div>`).join("");

  return `
    <div class="case-header">
      <div>
        <div class="case-id">${esc(a.account_id)}</div>
        <div class="case-tags">
          <span class="band ${bandClass(band)}">${band}</span>
          ${fraud}${ring}
        </div>
      </div>
      <div class="gauge">
        <svg viewBox="0 0 120 64">
          <path d="M10 54 A50 50 0 0 1 110 54" fill="none" stroke="var(--surface-3)" stroke-width="10" stroke-linecap="round"/>
          <path d="M10 54 A50 50 0 0 1 110 54" fill="none" stroke="var(--accent-400)" stroke-width="10" stroke-linecap="round"
                stroke-dasharray="${(Math.PI * 50).toFixed(1)}" stroke-dashoffset="${((1 - a.risk_score) * Math.PI * 50).toFixed(1)}"
                transform="rotate(180 60 54)"/>
        </svg>
        <div class="gauge-label">${a.risk_score.toFixed(3)}</div>
      </div>
    </div>
    <div class="kv">
      <div class="kv-row"><span class="k">Risk score</span><span class="v">${a.risk_score.toFixed(4)}</span></div>
      <div class="kv-row"><span class="k">Rank</span><span class="v">${rankTxt}</span></div>
      <div class="kv-row"><span class="k">Band</span><span class="v">${band}</span></div>
      <div class="kv-row"><span class="k">Degree</span><span class="v">${fmtIN(a.degree)}</span></div>
      <div class="kv-row"><span class="k">In-degree</span><span class="v">${fmtIN(a.in_degree ?? a.n_in_edges)}</span></div>
      <div class="kv-row"><span class="k">Out-degree</span><span class="v">${fmtIN(a.out_degree ?? a.n_out_edges)}</span></div>
    </div>
    <div class="section-h">Highest-risk neighbors</div>
    <div class="nbr-list">${neighbors || '<p class="muted">No neighbors.</p>'}</div>
    <div id="explain-zone"></div>
  `;
}

/* ---------- explain ---------- */
async function explain(aid) {
  const zone = $("explain-zone");
  if (!zone) return;
  zone.innerHTML = `<div class="skeleton" style="height:48px;margin-top:14px"></div>`;
  try {
    const e = await getJSON(`/api/explain/${encodeURIComponent(aid)}`);
    const ev = e.model_evidence || {};
    let drivers = "";
    if (ev.top_features && ev.top_features.length) {
      drivers = `<div class="explain-meta">drivers: ${ev.top_features.slice(0,3).map(f => esc(f.feature)).join(", ")}</div>`;
    }
    const source = e.source === "openai"
      ? `<div class="explain-meta">generated by OpenAI</div>`
      : "";
    zone.innerHTML = `<div class="explain-text">${esc(e.explanation)}</div>${drivers}${source}`;
  } catch (err) {
    zone.innerHTML = `<div class="explain-text">Explanation failed: ${esc(err.message)}</div>`;
  }
}

/* ---------- counterfactual (in canvas) ---------- */
async function showCounterfactual(aid) {
  const zone = $("explain-zone");
  if (!zone) return;
  zone.innerHTML = `<div class="skeleton" style="height:48px;margin-top:14px"></div>`;
  try {
    const cf = await getJSON(`/api/counterfactual/${encodeURIComponent(aid)}?k=3`);
    const dropped = cf.dropped_edges.map((e) =>
      `${esc(e.src.replace("acc_","a"))}→${esc(e.dst.replace("acc_","a"))} (risk ${e.risk != null ? e.risk.toFixed(2) : "—"})`
    ).join(", ");
    const low = 40, high = 70;
    const orig = cf.model_score_original * 100;
    const now  = cf.model_score_without_top_edges * 100;
    zone.innerHTML = `
      <div class="section-h">Counterfactual sensitivity</div>
      <div class="delta-bar">
        <span class="band ${bandClass(cf.band_served)}">${cf.band_served}</span>
        <div class="delta-track">
          <div class="marker marker-low" style="left:${low}%"></div>
          <div class="marker marker-high" style="left:${high}%"></div>
          <div class="score-bar-fill" style="position:absolute;left:0;top:0;bottom:0;width:${Math.max(0, Math.min(100, orig))}%;background:var(--accent-400);opacity:0.55"></div>
          <div class="score-bar-fill" style="position:absolute;left:0;top:0;bottom:0;width:${Math.max(0, Math.min(100, now))}%;background:var(--text);opacity:0.85"></div>
        </div>
        <span class="num" style="min-width:96px;text-align:right">${orig.toFixed(0)}→${now.toFixed(0)}</span>
      </div>
      <div class="explain-meta" style="margin-top:8px">drop top ${cf.k} edges: ${dropped}. ${esc(cf.caveat)}</div>
    `;
  } catch (err) {
    zone.innerHTML = `<div class="explain-text">Counterfactual failed: ${esc(err.message)}</div>`;
  }
}

/* ---------- case drawer ---------- */
async function openCase(aid) {
  const drawer = $("case-drawer");
  const doc = $("case-doc");
  drawer.hidden = false;
  requestAnimationFrame(() => {
    drawer.classList.add("open");
    $("drawer-backdrop").hidden = false;
    requestAnimationFrame(() => $("drawer-backdrop").classList.add("show"));
  });
  $("case-title").textContent = `Case file · ${aid}`;
  $("case-sub").textContent = "Generated deterministically · every figure is computed";
  doc.textContent = "Loading…";
  try {
    const r = await getJSON(`/api/case/${encodeURIComponent(aid)}`);
    doc.textContent = r.document;
    doc.dataset.aid = aid;
  } catch (err) {
    doc.textContent = `Failed to load case file: ${err.message}`;
  }
}
function closeCase() {
  const drawer = $("case-drawer");
  drawer.classList.remove("open");
  $("drawer-backdrop").classList.remove("show");
  setTimeout(() => {
    drawer.hidden = true;
    $("drawer-backdrop").hidden = true;
  }, 280);
}
$("case-close")?.addEventListener("click", closeCase);
$("drawer-backdrop")?.addEventListener("click", closeCase);
$("case-download")?.addEventListener("click", () => {
  const doc = $("case-doc");
  const aid = doc.dataset.aid || "case";
  const blob = new Blob([doc.textContent], { type: "text/markdown" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = `case-${aid}.md`;
  a.click(); URL.revokeObjectURL(url);
  toast("Case file downloaded");
});
$("case-copy")?.addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText($("case-doc").textContent);
    toast("Case file copied");
  } catch {
    toast("Copy failed");
  }
});

/* ---------- histogram ---------- */
async function loadDistribution() {
  const svg = $("hist-svg");
  const dist = await getJSON("/api/distribution?bins=20");
  const W = 720, H = 220, padL = 30, padR = 12, padT = 14, padB = 28;
  const innerW = W - padL - padR, innerH = H - padT - padB;
  const max = Math.max(1, ...dist.counts);
  const bw = innerW / dist.counts.length;
  const low = 0.4, high = 0.7;
  const xToPx = (x) => padL + (x * innerW);
  let inner = "";

  // axes
  inner += `<line class="hist-axis" x1="${padL}" y1="${H - padB}" x2="${W - padR}" y2="${H - padB}"/>`;
  inner += `<line class="hist-axis" x1="${padL}" y1="${padT}" x2="${padL}" y2="${H - padB}"/>`;

  // bars
  for (let i = 0; i < dist.counts.length; i++) {
    const h = (dist.counts[i] / max) * innerH;
    const x = padL + i * bw + 1;
    const y = H - padB - h;
    const left = dist.bins[i];
    const right = dist.bins[i + 1];
    const cls = right <= low ? "" : right <= high ? "medium" : "high";
    inner += `<rect class="hist-bar ${cls}" x="${x}" y="${y}" width="${bw - 2}" height="${h}">
      <title>${dist.counts[i]} accounts in [${left.toFixed(2)}, ${right.toFixed(2)})</title>
    </rect>`;
  }

  // thresholds
  const xLow = xToPx(low), xHigh = xToPx(high);
  inner += `<line class="hist-threshold" x1="${xLow}" y1="${padT}" x2="${xLow}" y2="${H - padB}"/>`;
  inner += `<text class="hist-threshold-label" x="${xLow + 4}" y="${padT + 10}">0.4 medium</text>`;
  inner += `<line class="hist-threshold" x1="${xHigh}" y1="${padT}" x2="${xHigh}" y2="${H - padB}"/>`;
  inner += `<text class="hist-threshold-label" x="${xHigh + 4}" y="${padT + 10}">0.7 high</text>`;

  // x labels
  inner += `<text class="hist-threshold-label" x="${padL}" y="${H - padB + 16}">${dist.bins[0].toFixed(2)}</text>`;
  inner += `<text class="hist-threshold-label" x="${padL + innerW / 2}" y="${H - padB + 16}" text-anchor="middle">risk score</text>`;
  inner += `<text class="hist-threshold-label" x="${W - padR}" y="${H - padB + 16}" text-anchor="end">${dist.bins[dist.bins.length - 1].toFixed(2)}</text>`;
  svg.innerHTML = inner;
}

/* ---------- assistant ---------- */
function appendBubble(role, html) {
  const thread = $("assistant-thread");
  const el = document.createElement("div");
  el.className = `bubble bubble-${role}`;
  el.innerHTML = html;
  thread.appendChild(el);
  thread.scrollTop = thread.scrollHeight;
  return el;
}
function askUserBubblesFor(question) {
  appendBubble("user", `<div class="bubble-text">${esc(question)}</div>`);
  const loading = appendBubble("assistant", `<div class="bubble-text muted">…</div>`);
  return loading;
}
function renderAnswer(a) {
  switch (a.kind) {
    case "account":     return renderAccount(a);
    case "ring":        return renderRing(a);
    case "top":         return renderTop(a);
    case "summary":     return renderSummary(a);
    case "case":        return renderCaseAnswer(a);
    case "counterfactual": return renderCounterfactual(a);
    default:            return `<div class="bubble-text">${esc(a.report || "")}</div>`;
  }
}
function renderAccount(a) {
  const f = a.facts || {};
  const band = f.risk_band || bandOf(f.risk_score);
  const open = (aid) => selectAccount(aid);
  return `
    <div class="ans-card">
      <div class="ans-card-row"><span class="k">Account</span><span class="mono">${esc(f.account_id || a.account_id)}</span></div>
      <div class="ans-card-row"><span class="k">Risk</span><span><span class="band ${bandClass(band)}">${band}</span> <span class="mono">${(f.risk_score ?? 0).toFixed(3)}</span></span></div>
      <div class="ans-card-row"><span class="k">Rank</span><span class="mono">${fmtIN(f.rank || 0)} of ${fmtIN(f.rank ? (state.summary?.n_accounts || 0) : 0)}</span></div>
      ${f.ring_id >= 0 ? `<div class="ans-card-row"><span class="k">Ring</span><span class="mono">${f.ring_id} · ${(f.ring_members || []).length} members · ${fmtIN(f.ring_edges || 0)} internal transfers</span></div>` : ""}
    </div>
    <div class="bubble-text">${esc(a.report || "")}</div>
    <div style="display:flex;gap:8px;flex-wrap:wrap">
      <button class="btn btn-ghost" data-aid="${esc(f.account_id || a.account_id)}">Open in canvas</button>
    </div>`;
}
function renderRing(a) {
  const f = a.facts || {};
  const members = (f.members || []).map((m) => `<span class="chip">${esc(m)}</span>`).join(" ");
  return `
    <div class="ans-card">
      <div class="ans-card-row"><span class="k">Ring</span><span class="mono">${f.ring_id ?? a.ring_id}</span></div>
      <div class="ans-card-row"><span class="k">Members</span><span>${members}</span></div>
      <div class="ans-card-row"><span class="k">Internal transfers</span><span class="mono">${fmtIN(f.n_internal_edges || 0)} · ${money(f.total_amount || 0)}</span></div>
      ${f.span_days != null ? `<div class="ans-card-row"><span class="k">Span</span><span class="mono">${Math.max(1, Math.round(f.span_days))} day(s)</span></div>` : ""}
    </div>
    <div class="bubble-text">${esc(a.report || "")}</div>
    <div style="display:flex;gap:8px;flex-wrap:wrap">
      <button class="btn btn-ghost" data-ring="${f.ring_id ?? a.ring_id}">Open in ring explorer</button>
    </div>`;
}
function renderTop(a) {
  const rows = (a.accounts || []).slice(0, 10).map((r) => `
    <div class="ans-list-item" data-aid="${esc(r.account_id)}">
      <span class="mono">${esc(r.account_id)}</span>
      <div class="ans-mini-bar"><span style="width:${(r.risk_score*100).toFixed(0)}%"></span></div>
      <span class="mono">${r.risk_score.toFixed(3)} · <span class="band ${bandClass(r.band)}">${r.band}</span></span>
    </div>`).join("");
  return `<div class="bubble-text">${esc(a.report || "")}</div><div class="ans-list">${rows}</div>`;
}
function renderSummary(a) {
  const f = a.facts || {};
  return `
    <div class="ans-card">
      <div class="ans-card-row"><span class="k">Accounts</span><span class="mono">${fmtIN(f.nodes || 0)}</span></div>
      <div class="ans-card-row"><span class="k">Transfers</span><span class="mono">${fmtIN(f.edges || 0)}</span></div>
      <div class="ans-card-row"><span class="k">Rings</span><span class="mono">${f.rings || 0}</span></div>
      <div class="ans-card-row"><span class="k">Fraud accounts</span><span class="mono">${fmtIN(f.fraud_nodes || 0)}</span></div>
      <div class="ans-card-row"><span class="k">High-risk (≥0.7)</span><span class="mono">${pct(f.risk_frac_high || 0)} of test set</span></div>
      <div class="ans-card-row"><span class="k">Model</span><span class="mono">${f.model || "?"}${f.cold_start ? " · cold-start fallback" : ""}</span></div>
    </div>
    <div class="bubble-text">${esc(a.report || "")}</div>`;
}
function renderCounterfactual(a) {
  const f = a.facts || {};
  const dropped = (f.dropped_edges || []).map((e) =>
    `${esc(e.src.replace("acc_","a"))}→${esc(e.dst.replace("acc_","a"))}`
  ).join(", ");
  return `
    <div class="ans-card">
      <div class="ans-card-row"><span class="k">Served score</span><span class="mono">${(f.score_served ?? 0).toFixed(3)} <span class="band ${bandClass(f.band_served)}">${f.band_served}</span></span></div>
      <div class="ans-card-row"><span class="k">Model, original</span><span class="mono">${(f.model_score_original ?? 0).toFixed(3)}</span></div>
      <div class="ans-card-row"><span class="k">Model, without top ${f.k}</span><span class="mono">${(f.model_score_without_top_edges ?? 0).toFixed(3)} <span class="band ${bandClass(f.band_without_top_edges)}">${f.band_without_top_edges}</span></span></div>
      <div class="ans-card-row"><span class="k">Δ model score</span><span class="mono">${(f.delta_model_score ?? 0).toFixed(4)}</span></div>
    </div>
    <div class="bubble-text muted">dropped: ${dropped}</div>
    <div class="explain-meta">${esc(f.caveat || "")}</div>
  `;
}
function renderCaseAnswer(a) {
  return `
    <div class="bubble-text">${esc(a.report || "")}</div>
    <div style="display:flex;gap:8px;flex-wrap:wrap">
      <button class="btn btn-primary" data-case="${esc(a.account_id)}">Open case file</button>
    </div>`;
}
function wireAnswerButtons(container) {
  container.querySelectorAll("[data-aid]").forEach((el) =>
    el.addEventListener("click", () => selectAccount(el.dataset.aid))
  );
  container.querySelectorAll("[data-ring]").forEach((el) =>
    el.addEventListener("click", () => {
      const sel = $("ring-select");
      sel.value = el.dataset.ring;
      sel.dispatchEvent(new Event("change"));
    })
  );
  container.querySelectorAll("[data-case]").forEach((el) =>
    el.addEventListener("click", () => openCase(el.dataset.case))
  );
}
async function ask(q) {
  if (!q || !q.trim()) return;
  askUserBubblesFor(q);
  const placeholder = $$(".bubble-assistant").slice(-1)[0];
  try {
    const a = await postJSON("/api/ask", { question: q });
    placeholder.innerHTML = renderAnswer(a);
    placeholder.classList.remove("empty");
    wireAnswerButtons(placeholder);
  } catch (e) {
    placeholder.innerHTML = `<div class="bubble-text">Error: ${esc(e.message)}</div>`;
  }
}

/* composer + chips */
$("composer")?.addEventListener("submit", (e) => {
  e.preventDefault();
  const input = $("composer-input");
  const q = input.value.trim();
  if (!q) return;
  input.value = "";
  ask(q);
});
$$(".chip-suggest").forEach((c) =>
  c.addEventListener("click", () => ask(c.dataset.q))
);

/* keyboard: ⌘K / Ctrl+K focuses composer, Esc closes drawer */
document.addEventListener("keydown", (e) => {
  const isMac = navigator.userAgent.includes("Mac");
  if ((isMac ? e.metaKey : e.ctrlKey) && e.key.toLowerCase() === "k") {
    e.preventDefault();
    $("composer-input")?.focus();
  }
  if (e.key === "Escape" && !$("case-drawer").hidden) {
    closeCase();
  }
});

/* ---------- boot ---------- */
initTheme();
Promise.all([
  loadSummary().catch((e) => console.error("summary:", e)),
  loadTop().catch((e) => console.error("top:", e)),
  loadDistribution().catch((e) => console.error("distribution:", e)),
]);
