// Dune AI Table — browser client. Polls /api/state, posts your moves to /api/act.
"use strict";

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

const FACTIONS = [["emperor", "EMP"], ["spacing_guild", "GLD"], ["bene_gesserit", "BG"], ["fremen", "FRE"]];

let S = null;            // last state
let lastKey = "";
let selectedCard = null; // card chosen from hand for an agent turn
let choiceSpace = null;  // space clicked that has several variants
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
      (S.thinking != null ? `${esc(pname(player(S.thinking)))} is thinking…`
        : S.active === S.seat ? "<b>Your move</b>" : `${esc(pname(act))} to move`) +
      ` · seed ${S.seed}`;
  renderBoard();
  renderConflict();
  renderContracts();
  renderBloodlines();
  renderPlayers();
  renderPurchases();
  renderHand(me);
  renderRow(me);
  renderTurn();
  renderLog();
}

function renderConflict() {
  const c = S.conflict;
  if (!c) { $("conflict").innerHTML = `<div class="muted">No conflict</div>`; return; }
  const max = Math.max(1, ...S.players.map((p) => p.strength));
  const bars = S.players.map((p) => `
    <div class="sbar"><span>${esc(p.colorName)}${p.you ? " (you)" : ""}</span>
      <div class="track"><div class="fill" style="width:${(100 * p.strength) / max}%;background:${p.color}"></div></div>
      <span>${p.inConflict} troops${p.worms ? ` +${p.worms} worm` : ""} · str ${p.strength}</span></div>`).join("");
  $("conflict").innerHTML = `
    ${c.img ? `<img src="${c.img}" data-img="${c.img}" alt="">` : ""}
    <div style="flex:1;min-width:0">
      <div class="panel-title">Conflict: ${esc(c.name)} <span class="muted">level ${c.level} · ${S.conflictsLeft} left</span>${c.icon ? ` ${iconChips([c.icon])}` : ""}</div>
      <div class="rewards">
        <div>🥇 ${esc(fmtObj(c.first))}</div><div>🥈 ${esc(fmtObj(c.second))}</div>${c.third ? `<div>🥉 ${esc(fmtObj(c.third))}</div>` : ""}
        ${c.location ? `<div class="muted">Location: ${esc(c.location)}</div>` : ""}
      </div>
      <div class="strength">${bars}</div>
    </div>`;
}

function fmtTrigger(t) {
  if (!t || !Object.keys(t).length) return "completes as soon as you take it";
  if (t.board_space) return `complete: send an agent to ${t.board_space}`;
  return "complete: " + fmtObj(t);
}
function renderContracts() {
  $("contracts").innerHTML = `<div class="panel-title">Contracts available <span class="muted">(take one at Accept Contract / Dutiful Service)</span></div>` +
    (S.contracts.length ? S.contracts.map((c) =>
      `<div class="c"><b>${esc(c.name)}</b> → <span>${esc(fmtObj(c.rewards))}</span><div class="trig">${esc(fmtTrigger(c.trigger))}</div></div>`).join("")
      : `<div class="muted">none face-up</div>`) +
    `<div class="muted" style="margin-top:6px">Shield Wall: ${S.shieldWall ? "intact" : "destroyed"}</div>`;
}

function renderBloodlines() {
  const bl = S.bl;
  $("blRow").style.display = bl ? "" : "none";
  if (!bl) return;
  $("techs").innerHTML = `<div class="panel-title">Tech market <span class="muted">(buy the top tile of a stack with spice when you send an agent to a green Landsraad space)</span></div>
    <div class="tech-grid">` + bl.techs.map((t) => t.top ? `
      <div class="tech">${t.top.img ? `<img src="${t.top.img}" data-img="${t.top.img}" alt="">` : ""}
        <div><div class="tn">${esc(t.top.name)}</div>
          <div class="tp">${t.top.price} spice${t.top.price !== t.top.cost ? ` (base ${t.top.cost})` : ""} · ${t.left - 1} more in stack</div>
          <div class="tt">${esc(t.top.text)}</div></div></div>`
      : `<div class="tech muted">Stack ${t.stack + 1} empty</div>`).join("") + `</div>`;
  const cmd = bl.commanders.map((c) => `<span class="cmd ${c.present ? "on" : "off"}" title="${c.present ? "commander still here" : "recruited"}">${esc(c.space)}</span>`).join("");
  const skills = bl.skills.map((k) => `
    <div class="skill">${k.img ? `<img src="${k.img}" data-img="${k.img}" alt="">` : ""}
      <div><div class="sn">${esc(k.name)}${k.heldBy.map((id) => `<span class="held" style="background:${player(id).color}" title="${esc(pname(player(id)))}"></span>`).join("")}</div>
      <div class="st">${esc(k.text)}</div></div></div>`).join("");
  const row = bl.skillRow.map((k) => `
    <div class="skill ${k.youHave ? "have" : ""}">${k.img ? `<img src="${k.img}" data-img="${k.img}" alt="">` : ""}
      <div><div class="sn">${esc(k.name)}${k.youHave ? ` <span class="muted">(you have it)</span>` : ""}</div>
      <div class="st">${esc(k.text)}</div></div></div>`).join("");
  $("sardaukar").innerHTML = `<div class="panel-title">Sardaukar commanders <span class="muted">(${bl.commanderCost} solari when you send an agent to one of these spaces)</span></div>
    <div class="cmd-list">${cmd}</div>
    <div class="panel-title small">Skill row <span class="muted">(take one you don't hold when recruiting from the board or with Plasteel Blades; the slot is refilled · ${bl.skillDeckLeft} tiles face-down)</span></div>
    <div class="skill-grid">${row || `<span class="muted">empty</span>`}</div>
    <details class="skill-ref"><summary class="muted">All 7 skills · who holds them</summary><div class="skill-grid">${skills}</div></details>`;
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
const BANNER_XY = { "Spice Refinery": [454, 291], "Arrakeen": [709, 291], "Imperial Basin": [969, 291] };
const COUNCIL_SEATS = [[340, 48], [384, 48], [430, 48], [476, 48]];
const SWORDMASTER_COIN = [538, 183];
const SHIELD_WALL = [880, 385];
const pctX = (x) => `${(100 * x) / BOARD_W}%`;
const pctY = (y) => `${(100 * y) / BOARD_H}%`;
const at = (x, y) => `left:${pctX(x)};top:${pctY(y)}`;

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
      ${s.makerSpice ? `<span class="maker" title="bonus spice">+${s.makerSpice}</span>` : ""}
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
  const banners = S.spaces.filter((s) => BANNER_XY[s.name] && s.controlledBy != null).map((s) =>
    `<span class="ctrl" style="${at(...BANNER_XY[s.name])};background:${player(s.controlledBy).color}" title="controlled by ${esc(pname(player(s.controlledBy)))}"></span>`).join("");
  const seats = S.players.filter((p) => p.councilor).map((p, k) =>
    `<span class="seat" style="${at(...COUNCIL_SEATS[k])};background:${p.color}" title="${esc(pname(p))}: High Council seat"></span>`).join("");
  const sm = S.players.some((p) => p.swordmaster) ? "" : `<span class="coin" style="${at(...SWORDMASTER_COIN)}">8</span>`;
  const wall = S.shieldWall ? "" : `<span class="wall-down" style="${at(...SHIELD_WALL)}" title="Shield Wall destroyed">✕</span>`;
  $("board").innerHTML = `<div class="bwrap"><img class="bimg" src="/static/board.png" alt="board" draggable="false">
    ${spaces}${banners}${seats}${spies}${sm}${wall}
    <div class="conflict" id="conflict" style="${at(706, 562)};width:${pctX(552)};height:${pctY(226)}"></div></div>`;
}

function renderPlayers() {
  $("players").innerHTML = `<div class="players-grid">` + S.players.map((p) => {
    const infl = FACTIONS.map(([f, tag]) =>
      `<div class="${p.alliances[f] ? "ally" : ""}" title="${f.replace("_", " ")}${p.alliances[f] ? " — alliance" : ""}">${tag} ${p.influence[f]}</div>`).join("");
    const hand = !p.you && p.hand ? `<div class="hidden-hand">${p.hand.map((c) => c.img
      ? `<img src="${c.img}" data-img="${c.img}" title="${esc(c.name)}">` : `<span class="chip">${esc(c.name)}</span>`).join("")}</div>` : "";
    const intr = !p.you && p.intrigues ? `<div class="sub">Intrigue: ${p.intrigues.map((x) => esc(x.name)).join(", ") || "none"}</div>` : "";
    const played = p.inPlay.length ? `<div class="sub">Played: ${p.inPlay.map((c) => esc(c.name)).join(", ")}</div>` : "";
    return `<div class="pl ${S.active === p.id && !S.gameOver ? "active" : ""}" style="border-left-color:${p.color}">
      <div class="hd"><span class="who">${esc(pname(p))}</span><span class="vp">${p.vp} VP</span></div>
      <div class="res"><span>💰 <b>${p.solari}</b></span><span>🟠 <b>${p.spice}</b></span><span>💧 <b>${p.water}</b></span>
        <span>🧍 <b>${p.agentsAvail}</b>/${p.agentsTotal}</span><span>🛡 <b>${p.garrison}</b></span><span title="Spies: in supply / on the board">🕵 <b>${p.spies}</b> / ${Object.values(S.spyPosts || {}).flat().filter((id) => id === p.id).length} on board</span></div>
      <div class="infl">${infl}</div>
      <div class="sub">Hand ${p.handSize} · deck ${p.deck} · intrigue ${p.intrigueCount}${p.swordmaster ? " · Swordmaster" : ""}${p.councilor ? " · High Council" : ""}${p.revealed ? " · revealed" : ""}</div>
      ${p.techs && p.techs.length ? `<div class="techs-mini">${p.techs.map((t) => t.img ? `<img src="${t.img}" data-img="${t.img}" title="${esc(t.name)}: ${esc(t.text)}">` : `<span class="chip">${esc(t.name)}</span>`).join("")}</div>` : ""}
      ${S.bl ? `<div class="sub">Commanders: ${p.cmdGarrison} garrison · ${p.cmdSupply} supply${p.cmdInConflict ? ` · ${p.cmdInConflict} in Conflict` : ""}${p.skills.length ? ` · skills: ${p.skills.map(esc).join(", ")}` : ""}</div>` : ""}
      ${p.contractsActive || p.contractsDone ? `<div class="sub">Contracts: ${p.contractsActive} active · ${p.contractsDone} done</div>` : ""}
      <div class="sub">Battle icons: ${p.battleIcons.length ? iconChips(p.battleIcons) : "none"}</div>
      ${played}${intr}${hand}
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
  $("handHint").textContent = myTurn && agentCards.size
    ? (selectedCard ? `· ${selectedCard}: pick a highlighted space` : "· click a card, then a highlighted space") : "";
  const intrActs = new Map(S.actions.filter((a) => a.type === "play_intrigue").map((a) => [a.intrigue, a.i]));
  $("intrigues").innerHTML = (me.intrigues || []).map((x) => intrActs.has(x.name)
    ? `<span class="chip act" data-act="${intrActs.get(x.name)}" title="Play it">${esc(x.name)} ▶</span>`
    : `<span class="chip">${esc(x.name)}</span>`).join("") || `<span class="muted">none</span>`;
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
    if (S.saved) h += `<div class="muted" style="margin-top:8px">Saved: ${esc(S.saved.log)}${S.saved.positions ? `<br>Your decisions: ${esc(S.saved.positions)}` : ""}${S.saved.training ? `<br>Training shard: ${esc(S.saved.training)}` : ""}</div>`;
    $("turn").innerHTML = h; return;
  }
  if (S.thinking != null) {
    const p = player(S.thinking);
    h += `<div class="thinking"><div class="spinner"></div>${esc(pname(p))} is thinking… ${S.thinkingFor}s</div>`;
    $("turn").innerHTML = h; return;
  }
  if (S.active !== S.seat) { $("turn").innerHTML = h + `<div class="muted">Waiting…</div>`; return; }

  h += `<div class="who-turn">Your move</div>`;
  if (S.choicePrompt) h += `<div class="prompt">${esc(S.choicePrompt)}</div>`;
  if (choiceSpace) {
    const vs = S.actions.filter((a) => a.type === "agent_turn" && a.card === selectedCard && a.space === choiceSpace);
    h += `<div class="group"><h5>${esc(selectedCard)} → ${esc(choiceSpace)}: choose</h5>` +
      vs.map((a) => `<button class="opt" data-act="${a.i}">${esc(a.label)}</button>`).join("") +
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
        `<details><summary>${esc(c)} (${as.filter((a) => a.card === c).length} options)</summary>` +
        as.filter((a) => a.card === c).map((a) => `<button class="opt" data-act="${a.i}">${esc(a.label)}</button>`).join("") +
        `</details>`).join("") + `</div>`;
    } else {
      h += `<div class="group"><h5>${title}</h5>` + as.map((a) => `<button class="opt" data-act="${a.i}">${esc(a.label)}</button>`).join("") + `</div>`;
    }
  }
  const rest = S.actions.filter((a) => !seen.has(a.i));
  if (rest.length) h += `<div class="group"><h5>Choose</h5>` + rest.map((a) => `<button class="opt" data-act="${a.i}">${esc(a.label)}</button>`).join("") + `</div>`;
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
      ${e.error ? `<span style="color:var(--bad)"> (${esc(e.error)})</span>` : ""}
      ${e.ai_pick && !e.agree ? `<div class="aipick">AI would have played: ${esc(e.ai_pick)}</div>` : ""}${why}</div>`;
  }).join("");
}

// ---------------------------------------------------------------- events
document.addEventListener("click", (ev) => {
  const t = ev.target.closest("[data-act],[data-card],[data-space],[data-cancel],.le.ai");
  if (!t || !S) return;
  if (t.dataset.act !== undefined) return act(+t.dataset.act);
  if (t.dataset.cancel) { choiceSpace = null; return render(); }
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

// ---------------------------------------------------------------- new game
function oppRows() {
  const seat = +$("seat").value;
  const colors = ["Red", "Green", "Blue", "Yellow"];
  const sel = options.opponents.map((o, i) => `<option value="${esc(o.spec)}" ${i === 0 ? "selected" : ""}>${esc(o.name)}: ${esc(o.note)}</option>`).join("");
  $("oppRows").innerHTML = [0, 1, 2, 3].filter((i) => i !== seat).map((i) =>
    `<label>${colors[i]} <select class="opp">${sel}</select></label>`).join("");
}
$("seat").addEventListener("change", oppRows);
$("newBtn").addEventListener("click", () => { oppRows(); $("newDlg").showModal(); });
$("newForm").addEventListener("submit", async (ev) => {
  if (ev.submitter && ev.submitter.value !== "ok") return;
  const opponents = [...document.querySelectorAll("#oppRows .opp")].map((s) => s.value);
  openLog.clear(); selectedCard = null; choiceSpace = null; lastKey = "";
  await post("/api/new", { seat: +$("seat").value, opponents, seed: $("seed").value.trim() || null, bloodlines: $("bloodlines").checked });
  poll(true);
});

// ---------------------------------------------------------------- polling
let timer = null;
async function poll(force) {
  clearTimeout(timer);
  try {
    const r = await fetch(`/api/state?reveal=${$("reveal").checked ? 1 : 0}`);
    const st = await r.json();
    const key = JSON.stringify([st.log && st.log.length, st.active, st.thinking, st.gameOver, st.error, st.actions && st.actions.length, $("reveal").checked, st.id, st.saved]);
    S = st;
    if (force || key !== lastKey) { lastKey = key; render(); }
    else if (st.thinking != null) renderTurn();        // just the timer
  } catch (e) {
    $("status").textContent = "Server not reachable: is ui/server.py running?";
  }
  const busy = S && !S.none && !S.gameOver && (S.thinking != null || S.active !== S.seat);
  timer = setTimeout(poll, busy ? 500 : 2000);
}

(async () => {
  options = await (await fetch("/api/options")).json();
  await poll(true);
  if (!S || S.none) { oppRows(); $("newDlg").showModal(); }
})();
