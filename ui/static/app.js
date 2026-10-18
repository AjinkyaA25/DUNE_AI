// Dune AI Table — browser client. Polls /api/state, posts your moves to /api/act.
"use strict";

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

const FACTIONS = [["emperor", "EMP"], ["spacing_guild", "GLD"], ["bene_gesserit", "BG"], ["fremen", "FRE"]];

let S = null;            // last state
let lastKey = "";
let selectedCard = null; // card chosen from hand for an agent turn
let choiceSpace = null;  // space clicked that has several variants
let choiceSpyPost = null; // observation post clicked that needs a recall choice
const openLog = new Set();
let options = { opponents: [] };

// ---------------------------------------------------------------- helpers
function pname(p) {
  if (!p) return "";
  return p.you ? `You (${p.colorName})` : `${p.colorName} · ${p.agentName}`;
}
function player(id) { return S.players.find((p) => p.id === id); }
function fmtObj(o) {
  if (!o) return "";
  if (typeof o === "string") return o;
  return Object.entries(o).map(([k, v]) => (v === 1 || v === true ? k : `${v} ${k}`)).join(", ").replace(/_/g, " ");
}
function fmtEffects(list) { return (list || []).map(fmtObj).filter(Boolean).join(" / "); }
function cardEl(c, cls = "", extra = "") {
  const inner = c.img
    ? `<img src="${c.img}" alt="${esc(c.name)}" loading="lazy">`
    : `<div class="txt"><b>${esc(c.name)}</b><br>${c.persuasion ? `${c.persuasion} persuasion<br>` : ""}${c.swords ? `${c.swords} swords` : ""}</div>`;
  const cost = c.cost != null && cls.includes("rowcard") ? `<span class="cost">${c.cost}</span>` : "";
  return `<div class="card ${cls}" data-img="${c.img || ""}" title="${esc(c.name)}" ${extra}>${inner}${cost}</div>`;
}
async function post(url, body) {
  const r = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  return r.json();
}
async function act(i) {
  selectedCard = null; choiceSpace = null;
  const r = await post("/api/act", { i });
  if (r.error) alertLine(r.error);
  poll(true);
}
function alertLine(msg) { $("status").textContent = "⚠ " + msg; }

// ---------------------------------------------------------------- render
function render() {
  if (!S || S.none) {
    $("status").textContent = "No game yet. Press “New game”.";
    return;
  }
  const me = player(S.seat);
  const act = S.active != null ? player(S.active) : null;
  $("status").innerHTML = S.gameOver
    ? `Game over · winner: <b>${esc(pname(player(S.winner)))}</b>`
    : `Round <b>${S.round}</b>/10 · ${esc(S.phase.replace(/_/g, " "))} · ` +
      (S.spectate ? (S.paused ? "Watching · <b>paused</b> · " : "Watching AI vs AI · ") : "") +
      (S.thinking != null ? `${esc(pname(player(S.thinking)))} is thinking…`
        : S.active === S.seat ? "<b>Your move</b>" : `${esc(pname(act))} to move`) +
      ` · seed ${S.seed}`;
  renderBoard();
  renderConflict();
  renderContracts();
  renderPlayers();
  renderPile();
  renderPurchases();
  renderHand(me);
  renderRow(me);
  renderEval();
  renderTurn();
  renderLog();
}

// ---------------------------------------------------------------- evaluation
function evalOn() { return $("evalOn").checked && S && S.evaluation; }
function pct(x) { return Math.round(100 * x) + "%"; }
function evDelta(d) {          // win-chance change, in percentage points
  const v = 100 * d, a = Math.abs(v);
  if (a < 0.5) return `<span class="evd flat">±0</span>`;
  return `<span class="evd ${v > 0 ? "up" : "down"}">${v > 0 ? "+" : "−"}${a < 10 ? a.toFixed(1) : Math.round(a)}</span>`;
}
function moveBadge(a) {
  if (!evalOn() || S.active !== S.seat) return "";
  const m = S.evaluation.moves || {}, x = m[a.i];
  if (x === undefined) return "";
  const best = Math.max(...Object.values(m));
  return `${x >= best - 1e-9 ? `<span class="evbest" title="Highest win chance of your moves">★</span>` : ""}${evDelta(x - S.evaluation.win[S.seat])}`;
}
function bestOf(as) {          // best badge among a group's options
  if (!evalOn()) return "";
  const m = S.evaluation.moves || {};
  const xs = as.map((a) => m[a.i]).filter((x) => x !== undefined);
  return xs.length ? ` <span class="muted">best</span> ${evDelta(Math.max(...xs) - S.evaluation.win[S.seat])}` : "";
}
function renderEval() {
  const el = $("evalp");
  if (!$("evalOn").checked) { el.hidden = true; return; }
  el.hidden = false;
  const E = S.evaluation;
  if (!E) {
    el.innerHTML = `<div class="panel-title">Evaluation</div><div class="muted">No evaluator yet. Train one with <code>python -m src.az.evaluator train</code>; the page picks it up automatically.</div>`;
    return;
  }
  const order = [...S.players].sort((a, b) => E.win[b.id] - E.win[a.id]);
  const bar = S.players.map((p) => `<div style="width:${100 * E.win[p.id]}%;background:${p.color}" title="${esc(pname(p))} ${pct(E.win[p.id])}"></div>`).join("");
  const rows = order.map((p) => `<div class="evrow ${p.you ? "me" : ""}"><span class="pd" style="background:${p.color}"></span>
      <span class="evn">${p.you ? "<b>You</b>" : esc(p.colorName)}</span><b>${pct(E.win[p.id])}</b>
      <span class="muted">→ ~${E.vp[p.id]} VP</span></div>`).join("");
  // win chance over the game (every move), one line per player
  const H = E.hist || [];
  let chart = "";
  if (H.length > 1) {
    const W = 300, Ht = 70, n0 = H[0].n, n1 = Math.max(H[H.length - 1].n, n0 + 1);
    const x = (n) => (W * (n - n0)) / (n1 - n0), y = (v) => Ht - Ht * v;
    const lines = S.players.map((p) => `<polyline fill="none" stroke="${p.color}" stroke-width="${p.you ? 2.4 : 1.2}" opacity="${p.you ? 1 : 0.7}"
        points="${H.map((h) => `${x(h.n).toFixed(1)},${y(h.win[p.id]).toFixed(1)}`).join(" ")}"/>`).join("");
    let ticks = "", lastR = null;
    for (const h of H) if (h.round !== lastR) { lastR = h.round; ticks += `<line x1="${x(h.n)}" x2="${x(h.n)}" y1="0" y2="${Ht}" class="evtick"/><text x="${x(h.n) + 2}" y="9" class="evlbl">R${h.round}</text>`; }
    chart = `<svg class="evchart" viewBox="0 0 ${W} ${Ht}" preserveAspectRatio="none">${ticks}<line x1="0" x2="${W}" y1="${y(0.25)}" y2="${y(0.25)}" class="evq"/>${lines}</svg>`;
  }
  const t = E.test || {};
  el.innerHTML = `<div class="panel-title">Evaluation <span class="muted">· from what you can see</span></div>
    <div class="evbar">${bar}</div>${rows}${chart}
    <div class="note muted">Win chance and expected final VP. Your move buttons show how each move changes your win chance (percentage points, ★ = best).
    ${t.acc ? `Held-out accuracy: picks the winner ${pct(t.acc)} of the time.` : ""}${E.trained ? ` Trained ${esc(E.trained)}.` : ""}</div>`;
}

function renderConflict() {
  const c = S.conflict;
  if (!c) { $("conflict").innerHTML = `<div class="muted">No conflict</div>${lastCombatLine()}`; return; }
  const rewards = `1st: ${fmtObj(c.first)} · 2nd: ${fmtObj(c.second)}${c.third ? ` · 3rd: ${fmtObj(c.third)}` : ""}`;
  // one corner per seat, as in the digital game: garrison inside the circle,
  // deployed units outside it toward the middle, strength in the corner header
  const corners = S.players.map((p, k) => {
    const pos = ["tl", "tr", "bl", "br"][k] || "tl";
    // the engine counts commanders as troops: split them back out
    const gCmd = p.cmdGarrison || 0, cCmd = p.cmdInConflict || 0;
    const gTr = Math.max(0, p.garrison - gCmd), cTr = Math.max(0, p.inConflict - cCmd);
    const str = p.strength || (p.inConflict ? 2 * p.inConflict + 3 * p.worms : 3 * p.worms);
    const cubes = (n, cls) => Array.from({ length: Math.min(n, 12) }, () => `<i class="${cls}" style="background:${p.color}"></i>`).join("");
    const unit = (cls, n) => `<span class="gu"><i class="${cls}" style="background:${p.color}"></i>${n}</span>`;
    return `<div class="fz ${pos}" title="${esc(pname(p))}: garrison ${gTr} troops${gCmd ? ` + ${gCmd} commander` : ""} · Conflict ${cTr} troops${cCmd ? ` + ${cCmd} commander` : ""}${p.worms ? ` + ${p.worms} sandworm` : ""} · strength ${str}${p.strength ? "" : " (units only, before swords)"}">
      <div class="fz-hd"><span class="fz-name" style="color:${p.color}">${esc(p.colorName)}${p.you ? " (you)" : ""}</span>
        <span class="hooks ${p.hooks ? "on" : "off"}" title="${p.hooks ? "Has Maker Hooks: can summon sandworms" : "No Maker Hooks yet"}">🪝</span>
        <span class="fz-str" style="border-color:${p.color}">⚔ ${str}</span></div>
      <div class="fz-body">
        <div class="garrison" style="border-color:${p.color}">${unit("cube", gTr)}${gCmd ? unit("dmd", gCmd) : ""}</div>
        <div class="deployed">${cubes(cTr, "cube")}${cTr > 12 ? `<span class="more">×${cTr}</span>` : ""}${cubes(cCmd, "dmd")}${"🪱".repeat(p.worms)}${cTr + cCmd + p.worms ? `<span class="dn">${cTr}${cCmd ? `+${cCmd}◆` : ""}</span>` : ""}</div>
      </div></div>`;
  }).join("");
  $("conflict").innerHTML = `
    <div class="fight">
      <div class="panel-title">${esc(c.name)} <span class="muted">level ${c.level} · ${S.conflictsLeft} left</span>${c.icon ? ` ${iconChips([c.icon])}` : ""}</div>
      ${c.img ? "" : `<div class="muted">${esc(rewards)}</div>`}
      <div class="arena">${corners}</div>
      ${lastCombatLine()}
    </div>
    ${c.img ? `<img src="${c.img}" data-img="${c.img}" alt="" title="${esc(rewards)}">` : ""}`;
}

// the Conflict just resolved (the panel above already shows the next one)
function lastCombatLine() {
  const L = S.lastCombat;
  if (!L) return "";
  const rows = L.rows.map((r) => {
    const p = player(r.pid);
    return `<b style="color:${p.color}">${esc(p.colorName)}${p.you ? " (you)" : ""}</b> str ${r.strength}: ${r.reward ? esc(fmtObj(r.reward)) : "nothing"}`;
  }).join(" · ");
  return `<div class="last-combat muted">Last combat, ${esc(L.name)}: ${rows || "nobody fought"}</div>`;
}

function fmtTrigger(t) {
  if (!t || !Object.keys(t).length) return "completes as soon as you take it";
  if (t.board_space) return `complete: send an agent to ${t.board_space}`;
  return "complete: " + fmtObj(t);
}
// face-up contracts are drawn in the board's right column (renderSide); this
// panel is hidden
function renderContracts() {
  $("contracts").hidden = true;
  $("contracts").innerHTML = `<div class="panel-title">Contracts available <span class="muted">(take one at Accept Contract / Dutiful Service)</span></div>` +
    (S.contracts.length ? S.contracts.map((c) =>
      `<div class="c"><b>${esc(c.name)}</b> → <span>${esc(fmtObj(c.rewards))}</span><div class="trig">${esc(fmtTrigger(c.trigger))}</div></div>`).join("")
      : `<div class="muted">none face-up</div>`) +
    `<div class="muted" style="margin-top:6px">Shield Wall: ${S.shieldWall ? "intact" : "destroyed"}</div>`;
}

// Bloodlines techs and Sardaukar skills, drawn right of the board (images only)
function renderSide() {
  const bl = S.bl;
  if (!bl) return "";
  const techs = bl.techs.map((t) => t.top ? `
    <div class="stech" title="${esc(t.top.name)} · ${t.top.price} spice${t.top.price !== t.top.cost ? ` (base ${t.top.cost})` : ""} · ${t.left - 1} more in stack">
      ${t.top.img ? `<img src="${t.top.img}" data-img="${t.top.img}" alt="${esc(t.top.name)}">` : `<span class="sname">${esc(t.top.name)}</span>`}
      ${t.top.price !== t.top.cost ? `<span class="sprice" title="your price">${t.top.price}</span>` : ""}${t.left > 1 ? `<span class="sleft">+${t.left - 1}</span>` : ""}</div>`
    : `<div class="stech empty"></div>`).join("");
  const icon = (k, cls, extra = "") => `<div class="sskill ${cls}" title="${esc(k.name)}${extra}">
      ${k.img ? `<img src="${k.img}" data-img="${k.img}" alt="${esc(k.name)}">` : `<span class="sname">${esc(k.name)}</span>`}`;
  const row = bl.skillRow.map((k) => icon(k, k.youHave ? "have" : "", k.youHave ? " (you have it)" : "") + `</div>`).join("");
  const held = bl.skills.map((k) => icon(k, k.heldBy.length ? "" : "nobody",
    k.heldBy.length ? `: ${k.heldBy.map((id) => pname(player(id))).join(", ")}` : ": nobody") +
    `<span class="sdots">${k.heldBy.map((id) => `<i style="background:${player(id).color}"></i>`).join("")}</span></div>`).join("");
  return `<div class="side">
    <div class="slabel">Tech</div><div class="stechs">${techs}</div>
    <div class="slabel" title="take one you don't hold when recruiting a commander · ${bl.skillDeckLeft} face-down">Skill row</div>
    <div class="sskills">${row || `<span class="muted">empty</span>`}</div>
    <div class="slabel" title="commander cost when you send an agent to a commander space">Commander ${bl.commanderCost}💰</div>
    <div class="slabel">Held skills</div><div class="sskills small">${held}</div>
    <div class="slabel" title="take one at Accept Contract / Dutiful Service">Contracts</div>
    <div class="scontracts">${(S.contracts || []).map((c) => `<div class="scontract" title="${esc(c.name)} → ${esc(fmtObj(c.rewards))} · ${esc(fmtTrigger(c.trigger))}">
      ${c.img ? `<img src="${c.img}" data-img="${c.img}" alt="${esc(c.name)}">` : `<span class="sname">${esc(c.name)}</span>`}</div>`).join("") || `<span class="muted">none face-up</span>`}</div></div>`;
}

function legalSpaces() {
  if (!selectedCard) return new Set();
  return new Set(S.actions.filter((a) => a.type === "agent_turn" && a.card === selectedCard).map((a) => a.space));
}

// Board picture (ui/static/board.png, built by ui/make_board_image.py): pixel
// boxes of each space and observation post, drawn over with the live state.
const BOARD_W = 1268, BOARD_H = 795;
const SPACE_BOX = {
  "Sardaukar": [2, 2, 218, 92], "Dutiful Service": [2, 98, 218, 186],
  "Heighliner": [2, 203, 218, 292], "Deliver Supplies": [2, 298, 218, 386],
  "Espionage": [2, 403, 218, 490], "Secrets": [2, 497, 218, 584],
  "Desert Tactics": [2, 603, 218, 690], "Fremkit": [2, 697, 218, 785],
  "High Council": [258, 20, 580, 116], "Imperial Privilege": [258, 132, 486, 230],
  "Swordmaster": [512, 132, 742, 230], "Assembly Hall": [754, 20, 982, 116],
  "Gather Support": [754, 132, 982, 230], "Shipping": [998, 20, 1228, 116],
  "Accept Contract": [998, 134, 1228, 230], "Spice Refinery": [388, 270, 614, 366],
  "Arrakeen": [643, 270, 871, 366], "Imperial Basin": [900, 270, 1130, 366],
  "Research Station": [328, 414, 558, 510], "Sietch Tabr": [328, 557, 558, 653],
  "Deep Desert": [606, 414, 836, 510], "Hagga Basin": [884, 414, 1112, 510],
};
const POST_XY = {
  "Emperor Post": [235, 100], "Spacing Guild Post": [235, 299], "Bene Gesserit Post": [235, 498],
  "Fremen Post": [235, 700], "Landsraad Post": [618, 92], "Green Post": [705, 107],
  "Shipping Post": [1245, 136], "Arrakeen Post": [655, 395], "Research Station Right Post": [575, 395],
  "Research Station Left Post": [575, 546], "Deep Desert Post": [703, 529],
  "Hagga Basin Post": [975, 529], "Imperial Basin Post": [1126, 410],
};
// where the board shows bonus spice on the maker spaces
const MAKER_XY = { "Deep Desert": [742, 485], "Hagga Basin": [1019, 485], "Imperial Basin": [1062, 322] };
const BANNER_XY = { "Spice Refinery": [454, 291], "Arrakeen": [709, 291], "Imperial Basin": [969, 291] };
const COUNCIL_SEATS = [[340, 48], [384, 48], [430, 48], [476, 48]];
const SWORDMASTER_COIN = [538, 183];
// one marker per player holding the Swordmaster (3rd agent), along the space
const SWORDMASTER_MARKS = [[606, 196], [642, 196], [678, 196], [714, 196]];
const SHIELD_WALL = [880, 385];
const pctX = (x) => `${(100 * x) / BOARD_W}%`;
const pctY = (y) => `${(100 * y) / BOARD_H}%`;
const at = (x, y) => `left:${pctX(x)};top:${pctY(y)}`;

// influence tracks, drawn left of the board beside each faction's two spaces
const TRACK_COLOR = { emperor: "#c9c9c9", spacing_guild: "#e0443c", bene_gesserit: "#b06ce0", fremen: "#4fb6ec" };
const TRACK_Y = { emperor: [2, 186], spacing_guild: [203, 386], bene_gesserit: [403, 584], fremen: [603, 785] };
function renderTracks() {
  return FACTIONS.map(([f]) => {
    const [y0, y1] = TRACK_Y[f];
    const ally = S.players.find((p) => p.alliances[f]);
    const rows = [6, 5, 4, 3, 2, 1, 0].map((lv) => {
      const cubes = S.players.filter((p) => Math.min(6, p.influence[f]) === lv).map((p) =>
        `<i class="tcube" style="background:${p.color}" title="${esc(pname(p))}: ${p.influence[f]} influence"></i>`).join("");
      const mark = lv === 4 ? `<span class="tmark" title="4+: alliance">★</span>` : lv === 2 ? `<span class="tmark" title="2+: 1 VP">VP</span>` : "";
      return `<div class="trow lv${lv}">${mark}${cubes}</div>`;
    }).join("");
    return `<div class="track-col" style="top:${pctY(y0)};height:${pctY(y1 - y0)};--fc:${TRACK_COLOR[f]}" title="${f.replace("_", " ")} influence">
      <span class="ally" style="${ally ? `background:${ally.color}` : ""}" title="${ally ? `alliance: ${esc(pname(ally))}` : "alliance not taken"}"></span>${rows}</div>`;
  }).join("");
}

function renderBoard() {
  const legal = legalSpaces();
  const cmdrs = new Set(S.bl ? S.bl.commanders.filter((c) => c.present).map((c) => c.space) : []);
  const spaces = S.spaces.filter((s) => SPACE_BOX[s.name]).map((s) => {
    const [x0, y0, x1, y1] = SPACE_BOX[s.name];
    const occ = s.occupant != null ? player(s.occupant) : null;
    const cls = ["bspace", legal.has(s.name) ? "legal" : "", selectedCard && !legal.has(s.name) ? "dim" : ""].join(" ");
    return `<div class="${cls}" data-space="${esc(s.name)}" title="${esc(s.name)}${occ ? ` · ${esc(pname(occ))}` : ""}"
      style="${at(x0, y0)};width:${pctX(x1 - x0)};height:${pctY(y1 - y0)}">
      ${occ ? `<span class="agent" style="background:${occ.color}"></span>` : ""}
      ${cmdrs.has(s.name) ? `<span class="cmdr-dot" title="Sardaukar commander available"></span>` : ""}
    </div>`;
  }).join("");
  // one circle per occupied post, split into equal slices when several Spies share it
  const spies = Object.entries(S.spyPosts || {}).filter(([post, ids]) => POST_XY[post] && ids.length).map(([post, ids]) => {
    const step = 360 / ids.length;
    const fill = ids.length === 1 ? player(ids[0]).color
      : `conic-gradient(${ids.map((id, k) => `${player(id).color} ${k * step}deg ${(k + 1) * step}deg`).join(", ")})`;
    return `<span class="spy" style="${at(...POST_XY[post])};background:${fill}"
      title="${esc(post)}: ${ids.map((id) => esc(pname(player(id)))).join(", ")}"></span>`;
  }).join("");
  // where you may place a Spy right now: glowing circles you click directly
  const spyTargets = [...new Set(S.actions.filter((a) => a.type === "resolve_spy" && POST_XY[a.post]).map((a) => a.post))]
    .map((post) => `<span class="spy-target ${post === choiceSpyPost ? "on" : ""}" data-spypost="${esc(post)}"
      style="${at(...POST_XY[post])}" title="Place your Spy: ${esc(post)}"></span>`).join("");
  const maker = S.spaces.filter((s) => MAKER_XY[s.name] && s.makerSpice).map((s) =>
    `<span class="maker" style="${at(...MAKER_XY[s.name])}" title="${esc(s.name)}: ${s.makerSpice} bonus spice">${s.makerSpice}</span>`).join("");
  const banners = S.spaces.filter((s) => BANNER_XY[s.name] && s.controlledBy != null).map((s) =>
    `<span class="ctrl" style="${at(...BANNER_XY[s.name])};background:${player(s.controlledBy).color}" title="controlled by ${esc(pname(player(s.controlledBy)))}"></span>`).join("");
  const seats = S.players.filter((p) => p.councilor).map((p, k) =>
    `<span class="seat" style="${at(...COUNCIL_SEATS[k])};background:${p.color}" title="${esc(pname(p))}: High Council seat"></span>`).join("");
  const sm = (S.players.some((p) => p.swordmaster) ? "" : `<span class="coin" style="${at(...SWORDMASTER_COIN)}">8</span>`) +
    S.players.filter((p) => p.swordmaster).map((p, k) =>
      `<span class="smark" style="${at(...SWORDMASTER_MARKS[k])};background:${p.color}" title="${esc(pname(p))}: has the Swordmaster (3rd agent)">⚔</span>`).join("");
  const wall = S.shieldWall ? "" : `<span class="wall-down" style="${at(...SHIELD_WALL)}" title="Shield Wall destroyed">✕</span>`;
  $("board").innerHTML = `<div class="bwrap"><img class="bimg" src="/static/board.png" alt="board" draggable="false">
    <div class="tracks">${renderTracks()}</div>${renderSide()}
    ${spaces}${maker}${banners}${seats}${spies}${spyTargets}${sm}${wall}
    <div class="conflict" id="conflict" style="${at(706, 562)};width:${pctX(552)};height:${pctY(226)}"></div></div>`;
}

function renderPlayers() {
  $("players").innerHTML = `<div class="players-grid">` + S.players.map((p) => {
    const infl = FACTIONS.map(([f, tag]) =>
      `<div class="${p.alliances[f] ? "ally" : ""}" title="${f.replace("_", " ")}${p.alliances[f] ? " — alliance" : ""}">${tag} ${p.influence[f]}</div>`).join("");
    const hand = !p.you && p.hand ? `<div class="hidden-hand">${p.hand.map((c) => c.img
      ? `<img src="${c.img}" data-img="${c.img}" title="${esc(c.name)}">` : `<span class="chip">${esc(c.name)}</span>`).join("")}</div>` : "";
    const intr = !p.you && p.intrigues ? (p.intrigues.length
      ? `<div class="intr-imgs">${p.intrigues.map((x) => x.img ? `<img src="${x.img}" data-img="${x.img}" title="${esc(x.name)}" alt="${esc(x.name)}">` : `<span class="chip">${esc(x.name)}</span>`).join("")}</div>`
      : `<div class="sub">Intrigue: none</div>`) : "";
    const L = p.leader;
    const leader = L ? `<div class="leader">${L.img ? `<img src="${L.img}" data-img="${L.img}" alt="${esc(L.name)}" title="${esc(L.name)}">` : ""}
      <div><div class="lname">${esc(L.name)}</div>${(L.status || []).map((t) => `<div class="lstat">${esc(t)}</div>`).join("")}
      ${L.nav ? `<div class="nav-slots">${L.nav.map((c) => c.img
        ? `<img src="${c.img}" data-img="${c.img}" class="${c.played ? "played" : ""}" title="Slot ${c.slot}${c.played ? " (played)" : ""}" alt="Navigation slot ${c.slot}">`
        : `<span class="nav-back" title="Slot ${c.slot}: face down">${c.slot}</span>`).join("")}</div>` : ""}</div></div>` : "";
    const rv = p.revealedThisRound ? `<div class="revealed-line" title="What this player revealed this round">Revealed: ${p.revealedThisRound.persuasion} persuasion · ${p.revealedThisRound.strength} strength</div>` : "";
    const unseenN = (p.unseen || []).reduce((t, x) => t + x[1], 0);
    const piles = `<div class="piles">
      <button class="pile-btn" data-pile="discard" data-pid="${p.id}">Discard <b>${p.discard.length}</b></button>
      <button class="pile-btn" data-pile="unseen" data-pid="${p.id}">${p.you ? "Deck" : "Hand or deck"} <b>${unseenN}</b></button></div>`;
    const bp = p.buyPower ? `<div class="sub" title="from public info: their owned cards minus discard and played cards">Reveal persuasion ~${p.buyPower.mean} · reach 5: ${Math.round(100 * p.buyPower.p["5"])}% · 7: ${Math.round(100 * p.buyPower.p["7"])}% · 9: ${Math.round(100 * p.buyPower.p["9"])}%</div>` : "";
    const played = p.inPlay.length ? `<div class="sub">Played: ${p.inPlay.map((c) => esc(c.name)).join(", ")}</div>` : "";
    return `<div class="pl ${S.active === p.id && !S.gameOver ? "active" : ""}" data-pid="${p.id}" title="Click for everything this player holds" style="border-left-color:${p.color}">
      <div class="hd"><span class="who">${esc(pname(p))}</span><span class="vp">${p.vp} VP</span></div>
      <div class="res"><span>💰 <b>${p.solari}</b></span><span>🟠 <b>${p.spice}</b></span><span>💧 <b>${p.water}</b></span>
        <span title="Agents this player can still send this round (out of their total)" style="font-weight:600">🧍 Agents left <b style="font-size:1.15em">${p.agentsAvail}</b> of ${p.agentsTotal}</span><span title="garrison: troops (commanders included)">🛡 <b>${p.garrison}</b></span><span title="Spies: in supply / on the board">🕵 <b>${p.spies}</b> / ${Object.values(S.spyPosts || {}).flat().filter((id) => id === p.id).length} on board</span></div>
      ${leader}${rv}
      <div class="infl">${infl}</div>
      ${p.topCard ? `<div class="sub topcard">Glowglobes · top of deck: <b data-img="${p.topCard.img || ""}">${esc(p.topCard.name)}</b></div>` : ""}
      <div class="sub">Hand ${p.handSize} · deck ${p.deck} · intrigue ${p.intrigueCount}${p.swordmaster ? " · Swordmaster" : ""}${p.councilor ? " · High Council" : ""}${p.revealed ? " · revealed" : ""}</div>
      ${p.techs && p.techs.length ? `<div class="owned"><span class="owned-lbl">Techs</span><div class="techs-mini">${p.techs.map((t) => t.img ? `<img src="${t.img}" data-img="${t.img}" title="${esc(t.name)}: ${esc(t.text)}">` : `<span class="chip">${esc(t.name)}</span>`).join("")}</div></div>` : ""}
      ${(p.contractList || []).length ? `<div class="owned"><span class="owned-lbl">Contracts</span><div class="contracts-mini">${p.contractList.map((c) => `<span class="cm ${c.done ? "done" : ""}" title="${esc(c.name)} · ${c.done ? "completed" : "active"}">${c.img ? `<img src="${c.img}" data-img="${c.img}" alt="${esc(c.name)}">` : `<span class="chip">${esc(c.name)}</span>`}${c.done ? `<i>✓</i>` : ""}</span>`).join("")}</div></div>` : ""}
      ${S.bl ? `<div class="sub">Commanders: ${p.cmdGarrison} garrison · ${p.cmdSupply} supply${p.cmdInConflict ? ` · ${p.cmdInConflict} in Conflict` : ""}${p.skills.length ? ` · skills: ${p.skills.map(esc).join(", ")}` : ""}</div>` : ""}
      ${S.bl && p.you && p.cmdSupply > 0 ? `<div class="sub cmd-hint">⚔ ${p.cmdSupply} commander${p.cmdSupply > 1 ? "s" : ""} in supply: re-recruit for ${S.bl.commanderCost} solari after any Agent turn</div>` : ""}
      <div class="sub">Battle icons: ${p.battleIcons.length ? iconChips(p.battleIcons) : "none"}</div>
      ${played}${bp}${piles}${intr}${hand}
    </div>`;
  }).join("") + `</div>`;
}

const ICON_CLS = { "Crysknife": "ic-knife", "Desert Mouse": "ic-mouse", "Ornithopter": "ic-thopter", "Wild": "ic-wild" };
function iconChips(list) {
  return list.map((i) => `<span class="bicon ${ICON_CLS[i] || ""}">${esc(i)}</span>`).join("");
}

function renderPurchases() {
  $("purchases").innerHTML = `<div class="panel-title">Cards bought <span class="muted">(by round; hover for the card)</span></div>
    <div class="buy-grid">` + S.players.map((p) => `
      <div class="buycol" style="border-top:3px solid ${p.color}">
        <div class="who">${esc(pname(p))} <span class="muted">· ${p.purchases.length}</span></div>
        ${p.purchases.length ? p.purchases.map((b) => `
          <div class="buy" data-img="${b.img || ""}"><span class="r">R${b.round}</span> ${esc(b.card)}${b.cost != null ? ` <span class="muted">(${b.cost})</span>` : ""}${b.free ? ` <span class="muted">free</span>` : ""}</div>`).join("")
          : `<div class="muted">nothing yet</div>`}
      </div>`).join("") + `</div>`;
}

function renderHand(me) {
  const myTurn = S.active === S.seat && !S.thinking;
  const agentCards = new Set(S.actions.filter((a) => a.type === "agent_turn").map((a) => a.card));
  $("hand").innerHTML = (me.hand || []).map((c) => {
    const cls = (agentCards.has(c.name) ? "playable " : "") + (selectedCard === c.name ? "selected" : "");
    return cardEl(c, cls, `data-card="${esc(c.name)}"`);
  }).join("") || `<span class="muted">empty</span>`;
  if (me.glowglobes)                            // Glowglobes: the top of your deck, always visible
    $("hand").innerHTML += `<div class="topdeck"><div class="topdeck-lbl">Top of deck<br><span class="muted">Glowglobes</span></div>` +
      (me.topCard ? cardEl(me.topCard, "") : `<span class="muted">deck empty: the next draw reshuffles your discard</span>`) + `</div>`;
  $("handHint").textContent = myTurn && agentCards.size
    ? (selectedCard ? `· ${selectedCard}: pick a highlighted space` : "· click a card, then a highlighted space") : "";
  const intrActs = new Map(S.actions.filter((a) => a.type === "play_intrigue").map((a) => [a.intrigue, a.i]));
  $("intrigues").innerHTML = (me.intrigues || []).map((x) => {
    const act = intrActs.has(x.name);
    if (!x.img) return act ? `<span class="chip act" data-act="${intrActs.get(x.name)}" title="Play it">${esc(x.name)} ▶</span>`
      : `<span class="chip">${esc(x.name)}</span>`;
    return `<span class="intr-card ${act ? "act" : ""}" ${act ? `data-act="${intrActs.get(x.name)}"` : ""} title="${esc(x.name)}${act ? " (click to play)" : ""}">
      <img src="${x.img}" data-img="${x.img}" alt="${esc(x.name)}">${act ? `<span class="play">Play ▶</span>` : ""}</span>`;
  }).join("") || `<span class="muted">none</span>`;
  $("inplay").innerHTML = me.inPlay.map((c) => cardEl(c)).join("") || `<span class="muted">nothing yet</span>`;
}

function renderRow(me) {
  const buy = new Map(S.actions.filter((a) => a.type === "acquire_card").map((a) => [a.buy, a.i]));
  $("row").innerHTML = S.row.map((c) => cardEl(c, "rowcard " + (buy.has(c.name) ? "buyable" : ""),
    buy.has(c.name) ? `data-act="${buy.get(c.name)}"` : "")).join("");
  const res = new Map(S.actions.filter((a) => a.type === "acquire_reserve").map((a) => [a.buy, a.i]));
  $("reserve").innerHTML = [["prepare_the_way", "Prepare the Way"], ["spice_must_flow", "The Spice Must Flow"]].map(([k, n]) =>
    res.has(k) ? `<span class="chip act" data-act="${res.get(k)}">Buy ${n} (${S.reserve[k]} left)</span>`
      : `<span class="chip">${n} · ${S.reserve[k]} left</span>`).join("");
  $("persuasion").textContent = me.persuasion ? `· you have ${me.persuasion} persuasion` : "";
}

const GROUPS = [
  ["agent_turn", "Agent turns"], ["reveal_turn", "Reveal"], ["acquire_card", "Buy"], ["acquire_reserve", "Buy (reserve)"],
  ["end_reveal", "End turn"], ["play_intrigue", "Intrigue"], ["combat_pass", "Combat"],
];
function renderTurn() {
  let h = "";
  if (S.error) h += `<pre style="color:var(--bad);white-space:pre-wrap">${esc(S.error)}</pre>`;
  if (S.gameOver) {
    const ranked = [...S.players].sort((a, b) => b.vp - a.vp);
    h += `<div class="over">Game over</div>` + ranked.map((p) =>
      `<div>${p.id === S.winner ? "🏆 " : ""}${esc(pname(p))}: <b>${p.vp} VP</b></div>`).join("");
    if (S.saved && S.saved.log) h += `<div class="muted" style="margin-top:8px">Saved: ${esc(S.saved.log)}${S.saved.positions ? `<br>Your decisions: ${esc(S.saved.positions)}` : ""}${S.saved.training ? `<br>Training shard: ${esc(S.saved.training)}` : ""}</div>`;
    $("turn").innerHTML = h; return;
  }
  if (S.spectate) h += `<div class="watchbar"><b>${S.replay ? "Replay" : "Watching"}</b> · ${S.replay ? `recorded self-play game (final VP ${(S.replay.vp || []).join("/")}, ${S.replay.rounds} rounds)` : "every seat is an AI"}
    <button class="btn ghost" id="pauseBtn">${S.paused ? "▶ Resume" : "⏸ Pause"}</button></div>`;
  if (S.thinking != null) {
    const p = player(S.thinking);
    h += `<div class="thinking"><div class="spinner"></div>${esc(pname(p))} is thinking… ${S.thinkingFor}s</div>`;
    $("turn").innerHTML = h; return;
  }
  if (S.spectate || S.active !== S.seat) { $("turn").innerHTML = h + `<div class="muted">${S.paused ? "Paused" : "Waiting…"}</div>`; return; }

  h += `<div class="who-turn">Your move</div>`;
  if (S.choicePrompt) h += `<div class="prompt">${esc(S.choicePrompt)}</div>`;
  if (choiceSpyPost) {
    const vs = S.actions.filter((a) => a.type === "resolve_spy" && a.post === choiceSpyPost);
    if (vs.length) {
      h += `<div class="group"><h5>Spy → ${esc(choiceSpyPost)}: choose</h5>` +
        vs.map((a) => `<button class="opt" data-act="${a.i}">${esc(a.label)}${moveBadge(a)}</button>`).join("") +
        `<button class="opt" data-cancel="1">← back</button></div>`;
      $("turn").innerHTML = h; return;
    }
    choiceSpyPost = null;
  }
  if (S.actions.some((a) => a.type === "resolve_spy" && POST_XY[a.post]))
    h += `<div class="muted spy-hint">Click a glowing circle on the board to place your Spy, or pick below.</div>`;
  if (choiceSpace) {
    const vs = S.actions.filter((a) => a.type === "agent_turn" && a.card === selectedCard && a.space === choiceSpace);
    h += `<div class="group"><h5>${esc(selectedCard)} → ${esc(choiceSpace)}: choose</h5>` +
      vs.map((a) => `<button class="opt" data-act="${a.i}">${esc(a.label)}${moveBadge(a)}</button>`).join("") +
      `<button class="opt" data-cancel="1">← back</button></div>`;
    $("turn").innerHTML = h; return;
  }
  const seen = new Set();
  for (const [type, title] of GROUPS) {
    const as = S.actions.filter((a) => a.type === type);
    as.forEach((a) => seen.add(a.i));
    if (!as.length) continue;
    if (type === "agent_turn") {
      const cards = [...new Set(as.map((a) => a.card))];
      h += `<div class="group"><h5>${title}: pick a card in your hand, or:</h5>` + cards.map((c) =>
        `<details><summary>${esc(c)} (${as.filter((a) => a.card === c).length} options)${bestOf(as.filter((a) => a.card === c))}</summary>` +
        as.filter((a) => a.card === c).map((a) => `<button class="opt" data-act="${a.i}">${esc(a.label)}${moveBadge(a)}</button>`).join("") +
        `</details>`).join("") + `</div>`;
    } else {
      h += `<div class="group"><h5>${title}</h5>` + as.map((a) => `<button class="opt" data-act="${a.i}">${esc(a.label)}${moveBadge(a)}</button>`).join("") + `</div>`;
    }
  }
  const rest = S.actions.filter((a) => !seen.has(a.i));
  const pics = rest.filter((a) => a.img);            // Navigation cards / Kota's Tech: pick by picture
  if (pics.length) h += `<div class="group"><div class="opt-pics">` + pics.map((a) =>
    `<button class="opt opt-pic" data-act="${a.i}" title="${esc(a.label)}"><img src="${a.img}" alt="">
      <span>${esc(a.label)}${moveBadge(a)}</span></button>`).join("") + `</div></div>`;
  const plain = rest.filter((a) => !a.img);
  if (plain.length) h += `<div class="group"><h5>Choose</h5>` + plain.map((a) => `<button class="opt" data-act="${a.i}">${esc(a.label)}${moveBadge(a)}</button>`).join("") + `</div>`;
  $("turn").innerHTML = h;
}

function renderLog() {
  const entries = [...S.log].reverse();
  $("log").innerHTML = entries.map((e) => {
    const p = player(e.pid);
    const ai = !!e.explain;
    let why = "";
    if (ai) {
      const vals = e.explain.rows.map((r) => r.value);
      const lo = Math.min(...vals), hi = Math.max(...vals), span = hi - lo || 1;
      const isSearch = e.explain.kind === "search";
      why = `<div class="why"><table>` + e.explain.rows.map((r) => `
        <tr class="${r.chosen ? "chosen" : ""}"><td>${r.chosen ? "✔" : ""}</td><td>${esc(r.label)}</td>
          <td style="width:70px"><div class="bar"><div style="width:${Math.max(4, (100 * (r.value - lo)) / span)}%"></div></div></td>
          <td style="text-align:right">${isSearch ? Math.round(100 * r.value) + "%" : r.value}</td></tr>`).join("") +
        `</table><div class="note">${esc(e.explain.note)}${e.secs ? ` · ${e.secs}s` : ""}</div></div>`;
    }
    return `<div class="le ${ai ? "ai" : ""} ${openLog.has(e.n) ? "open" : ""}" data-n="${e.n}">
      <span class="r">R${e.round}</span><span class="pd" style="background:${p ? p.color : "#666"}"></span>
      <span class="${ai ? "has-why" : ""}">${p && p.you ? "<b>You</b>" : esc(p ? p.colorName : "?")}: ${esc(e.label)}</span>
      ${$("evalOn").checked && e.ev !== undefined ? evDelta(e.ev) : ""}
      ${e.error ? `<span style="color:var(--bad)"> (${esc(e.error)})</span>` : ""}
      ${e.ai_pick && !e.agree ? `<div class="aipick">AI would have played: ${esc(e.ai_pick)}</div>` : ""}${why}</div>`;
  }).join("");
}

// ---------------------------------------------------------------- events
document.addEventListener("click", async (ev) => {
  if (ev.target.id === "pauseBtn" && S) {
    await post("/api/pause", { paused: !S.paused });
    return poll(true);
  }
}, true);
document.addEventListener("click", (ev) => {
  const t = ev.target.closest("[data-act],[data-card],[data-space],[data-cancel],[data-spypost],.le.ai");
  if (!t || !S) return;
  if (t.dataset.act !== undefined) { choiceSpyPost = null; return act(+t.dataset.act); }
  if (t.dataset.cancel) { choiceSpace = null; choiceSpyPost = null; return render(); }
  if (t.dataset.spypost !== undefined) {
    const vs = S.actions.filter((a) => a.type === "resolve_spy" && a.post === t.dataset.spypost);
    if (vs.length === 1) { choiceSpyPost = null; return act(vs[0].i); }
    choiceSpyPost = t.dataset.spypost;          // none in supply: which Spy comes back?
    return render();
  }
  if (t.dataset.card !== undefined) {
    if (!S.actions.some((a) => a.type === "agent_turn" && a.card === t.dataset.card)) return;
    selectedCard = selectedCard === t.dataset.card ? null : t.dataset.card;
    choiceSpace = null;
    return render();
  }
  if (t.dataset.space !== undefined && t.classList.contains("legal")) {
    const vs = S.actions.filter((a) => a.type === "agent_turn" && a.card === selectedCard && a.space === t.dataset.space);
    if (vs.length === 1) return act(vs[0].i);
    choiceSpace = t.dataset.space;
    return render();
  }
  if (t.classList.contains("le")) {
    const n = +t.dataset.n;
    openLog.has(n) ? openLog.delete(n) : openLog.add(n);
    t.classList.toggle("open");
  }
});

// ---------------------------------------------------------------- pile viewer
let pileView = null;   // {pid, kind} while the viewer is open
function renderPile() {
  if (!pileView || !S || S.none) return;
  const p = player(pileView.pid);
  if (!p) return;
  if (pileView.kind === "player") return renderPlayerDetail(p);
  $("pileBody").classList.remove("detail");
  let title, items;
  if (pileView.kind === "discard") {
    title = `${pname(p)}: discard pile (${p.discard.length})`;
    const m = new Map();
    p.discard.forEach((c) => { const e = m.get(c.name) || { c, n: 0 }; e.n++; m.set(c.name, e); });
    items = [...m.values()].sort((x, y) => y.n - x.n || x.c.name.localeCompare(y.c.name));
  } else {
    const u = p.unseen || [];
    title = p.you ? `Your deck (${u.reduce((t, x) => t + x[1], 0)}, order hidden)`
      : `${pname(p)}: in hand (${p.handSize}) or deck (${p.deck})`;
    items = u.map(([name, n, img]) => ({ c: { name, img }, n }));
  }
  $("pileTitle").textContent = title;
  $("pileBody").innerHTML = items.length ? items.map(({ c, n }) => `<div class="pile-card">${cardEl(c, "", "")}${n > 1 ? `<span class="pile-n">×${n}</span>` : ""}</div>`).join("")
    : `<div class="muted">empty</div>`;
}
// Click a player card: everything that player holds (hand and intrigues are
// shown for you, in watch mode, or with "Show AI hands")
function renderPlayerDetail(p) {
  const sec = (title, body) => `<section class="pd-sec"><h4>${title}</h4><div class="pd-row">${body}</div></section>`;
  const cards = (list) => list.length ? list.map((c) => `<div class="pile-card">${cardEl(c)}</div>`).join("") : `<span class="muted">none</span>`;
  const hidden = `<span class="muted">hidden: tick “Show AI hands”, or watch an AI-only game</span>`;
  const h = [];
  h.push(sec(`Hand (${p.handSize})`, p.hand ? cards(p.hand) : hidden));
  h.push(sec(`Intrigues (${p.intrigueCount})`, p.intrigues ? cards(p.intrigues) : hidden));
  h.push(sec("Battle icons", p.battleIcons.length ? iconChips(p.battleIcons) : `<span class="muted">none</span>`));
  h.push(sec(`Techs (${(p.techs || []).length})`, (p.techs || []).length
    ? p.techs.map((t) => t.img ? `<img class="pd-tech" src="${t.img}" data-img="${t.img}" title="${esc(t.name)}">` : `<span class="chip">${esc(t.name)}</span>`).join("")
    : `<span class="muted">none</span>`));
  const cl = p.contractList || [];
  h.push(sec(`Contracts (${cl.filter((c) => !c.done).length} active · ${cl.filter((c) => c.done).length} done)`, cl.length
    ? cl.map((c) => `<span class="cm ${c.done ? "done" : ""}" title="${esc(c.name)} · ${c.done ? "completed" : "active"}">${c.img ? `<img class="pd-contract" src="${c.img}" data-img="${c.img}">` : `<span class="chip">${esc(c.name)}</span>`}${c.done ? "<i>✓</i>" : ""}</span>`).join("")
    : `<span class="muted">none</span>`));
  if (S.bl) h.push(sec(`Sardaukar: ${p.cmdGarrison} in garrison · ${p.cmdSupply} in supply${p.cmdInConflict ? ` · ${p.cmdInConflict} in the Conflict` : ""}`,
    (p.skillImgs || []).length ? p.skillImgs.map(([k, img]) => img ? `<img class="pd-skill" src="${img}" data-img="${img}" title="${esc(k)}">` : `<span class="chip">${esc(k)}</span>`).join("")
      : `<span class="muted">no skills</span>`));
  h.push(sec(`In play this round (${p.inPlay.length})`, cards(p.inPlay)));
  $("pileTitle").innerHTML = `${esc(pname(p))} · <b>${p.vp} VP</b> · 💰${p.solari} 🟠${p.spice} 💧${p.water}${p.swordmaster ? " · Swordmaster" : ""}${p.councilor ? " · High Council" : ""}`;
  $("pileBody").classList.add("detail");
  $("pileBody").innerHTML = h.join("");
}
document.addEventListener("click", (ev) => {
  const b = ev.target.closest(".pile-btn");
  const card = !b && ev.target.closest(".players .pl");
  if (!b && !card) return;
  if (card && ev.target.closest("[data-act],button,a")) return;
  pileView = b ? { pid: +b.dataset.pid, kind: b.dataset.pile } : { pid: +card.dataset.pid, kind: "player" };
  renderPile();
  $("pileDlg").showModal();
});
$("pileDlg").addEventListener("close", () => { pileView = null; });
$("pileDlg").addEventListener("click", (ev) => { if (ev.target === $("pileDlg") || ev.target.closest(".pile-close")) $("pileDlg").close(); });

const zoom = $("zoom");
document.addEventListener("mousemove", (ev) => {
  const t = ev.target.closest("[data-img]");
  const src = t && t.dataset.img;
  if (!src) { zoom.style.display = "none"; return; }
  if (zoom.dataset.src !== src) { zoom.innerHTML = `<img src="${src}">`; zoom.dataset.src = src; }
  zoom.style.display = "block";
  const x = ev.clientX + 260 > innerWidth ? ev.clientX - 255 : ev.clientX + 15;
  const y = Math.min(ev.clientY + 10, innerHeight - 345);
  zoom.style.left = x + "px"; zoom.style.top = y + "px";
});

$("reveal").addEventListener("change", () => poll(true));
try { $("evalOn").checked = localStorage.getItem("evalOn") === "1"; } catch (e) {}
$("evalOn").addEventListener("change", () => {
  try { localStorage.setItem("evalOn", $("evalOn").checked ? "1" : "0"); } catch (e) {}
  render();
});

// ---------------------------------------------------------------- new game
async function loadReplays() {
  const r = await (await fetch("/api/replays")).json();
  const g = r.games || [];
  $("replayPick").innerHTML = g.length ? g.map((x) => `<option value="${esc(x.file)}|${x.line}">${esc(x.run)} · gen ${esc(x.gen)} · round ${x.rounds} · VP ${(x.vp || []).join("/")}${x.styles ? " · " + x.styles.join("/") : ""}</option>`).join("")
    : `<option value="">no recorded games yet (training runs write them to models_*/games/)</option>`;
}
function oppRows() {
  const seat = +$("seat").value;
  const colors = ["Red", "Green", "Yellow", "Blue"];
  const sel = options.opponents.map((o, i) => `<option value="${esc(o.spec)}" ${i === 0 ? "selected" : ""}>${esc(o.name)}: ${esc(o.note)}</option>`).join("");
  $("paceRow").hidden = seat >= 0;
  $("replayRow").hidden = seat !== -2;
  if (seat === -2) { loadReplays(); $("oppRows").innerHTML = ""; return; }
  if (options.opponents.length === 1) {       // one bot on offer: no choice to make
    $("oppRows").innerHTML = `<p class="muted">Opponents: ${esc(options.opponents[0].name)}</p>`;
    return;
  }
  $("oppRows").innerHTML = [0, 1, 2, 3].filter((i) => i !== seat).map((i) =>
    `<label>${colors[i]} <select class="opp">${sel}</select></label>`).join("");
}
function leaderRows() {
  const on = $("useLeaders").checked, seat = +$("seat").value;
  const bl = $("bloodlines").checked;
  $("leaderRow").hidden = !on || seat < 0;
  const list = (options.leaders || []).filter((l) => bl || (options.upLeaders || []).includes(l.name));
  const cur = $("leaderPick").value;
  $("leaderPick").innerHTML = `<option value="">Random</option>` + list.map((l) =>
    `<option value="${esc(l.name)}" ${l.name === cur ? "selected" : ""}>${esc(l.name)}${l.complexity ? " " + "◆".repeat(l.complexity) : ""}</option>`).join("");
  const pick = $("leaderPick").value;
  const tile = (name, img, label, extra) => `<button type="button" class="leader-tile${name === pick ? " on" : ""}"
    data-leader="${esc(name)}" role="option" aria-selected="${name === pick}" title="${esc(label)}">
    ${img ? `<img src="${img}" alt="" loading="lazy">` : `<span class="leader-tile-blank">?</span>`}
    <span>${esc(label)}${extra || ""}</span></button>`;
  $("leaderGrid").innerHTML = tile("", null, "Random") + list.map((l) =>
    tile(l.name, l.img, l.name, l.complexity ? ` <i>${"◆".repeat(l.complexity)}</i>` : "")).join("");
  $("leaderPicked").textContent = pick || "Random";
  const l = list.find((x) => x.name === pick);
  $("leaderPreview").innerHTML = on && seat >= 0 && l ? `${l.img ? `<img src="${l.img}" alt="${esc(l.name)}">` : ""}
    <div><b>${esc(l.name)}</b><p>${esc(l.passive || "")}</p><p><b>Signet Ring:</b> ${esc(l.signet || "")}</p></div>` : "";
}
$("useLeaders").addEventListener("change", leaderRows);
$("bloodlines").addEventListener("change", leaderRows);
$("leaderPick").addEventListener("change", leaderRows);
$("leaderGrid").addEventListener("click", (ev) => {
  const t = ev.target.closest(".leader-tile");
  if (!t) return;
  $("leaderPick").value = t.dataset.leader;
  leaderRows();
});
$("seat").addEventListener("change", () => { oppRows(); leaderRows(); });
$("newBtn").addEventListener("click", () => { oppRows(); leaderRows(); $("newDlg").showModal(); });
$("newForm").addEventListener("submit", async (ev) => {
  if (ev.submitter && ev.submitter.value !== "ok") return;
  const opponents = [...document.querySelectorAll("#oppRows .opp")].map((s) => s.value);
  openLog.clear(); selectedCard = null; choiceSpace = null; lastKey = "";
  const seatV = +$("seat").value;
  let replay = null;
  if (seatV === -2) {
    const v = $("replayPick").value;
    if (!v) return;
    const [file, line] = v.split("|");
    replay = { file, line: +line };
  }
  await post("/api/new", { seat: seatV === -2 ? -1 : seatV, opponents, seed: $("seed").value.trim() || null, bloodlines: $("bloodlines").checked, pace: +$("pace").value || 0, replay,
    leaders: $("useLeaders").checked, leader: $("leaderPick").value || null });
  poll(true);
});

// ---------------------------------------------------------------- polling
let timer = null;
async function poll(force) {
  clearTimeout(timer);
  try {
    const r = await fetch(`/api/state?reveal=${$("reveal").checked ? 1 : 0}`);
    const st = await r.json();
    const key = JSON.stringify([st.log && st.log.length, st.active, st.thinking, st.gameOver, st.error, st.actions && st.actions.length, $("reveal").checked, st.id, st.saved, st.paused, st.evaluation && st.evaluation.trained, st.evaluation && Object.keys(st.evaluation.moves || {}).length]);
    S = st;
    if (force || key !== lastKey) { lastKey = key; render(); }
    else if (st.thinking != null) renderTurn();        // just the timer
  } catch (e) {
    $("status").textContent = "Server not reachable: is ui/server.py running?";
  }
  const busy = S && !S.none && !S.gameOver && (S.spectate ? !S.paused : (S.thinking != null || S.active !== S.seat));
  timer = setTimeout(poll, busy ? 500 : 2000);
}

(async () => {
  options = await (await fetch("/api/options")).json();
  if (options.public) {                         // website: just you vs the Best AI
    for (const id of ["reveal", "evalOn"]) { $(id).checked = false; $(id).closest("label").hidden = true; }
    for (const o of [...$("seat").options]) if (+o.value < 0) o.remove();
  }
  await poll(true);
  if (!S || S.none) { oppRows(); leaderRows(); $("newDlg").showModal(); }
})();
