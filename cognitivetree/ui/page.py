"""Embedded single-file client for the streaming interface.

The page is served from memory with zero external assets so the UI works on
air-gapped hosts and inside restricted networks. Rendering is intentionally
plain: a phase log on the left, the live thought tree on the right, and the
accepted solution below once the run terminates.
"""

PAGE_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CognitiveTree-AI — Live Reasoning Stream</title>
<style>
  :root {
    --bg: #0f1419; --panel: #161c24; --border: #2a3441; --text: #d6dde6;
    --dim2: #77808c; --accent: #4da3ff; --ok: #3ecf8e;
    --bad: #ff6b6b; --warn: #f0b429; --mono: "Cascadia Code", Consolas, monospace;
  }
  * { box-sizing: border-box; margin: 0; }
  body { background: var(--bg); color: var(--text); font-family: var(--mono);
         font-size: 13px; height: 100vh; display: flex; flex-direction: column; }
  header { padding: 10px 16px; border-bottom: 1px solid var(--border);
           display: flex; align-items: center; gap: 12px; }
  header h1 { font-size: 14px; font-weight: 600; }
  #status { padding: 2px 10px; border-radius: 10px; font-size: 12px;
            background: var(--border); }
  #status.running   { color: var(--accent); }
  #status.succeeded { color: var(--ok); }
  #status.exhausted { color: var(--warn); }
  #status.failed    { color: var(--bad); }
  main { flex: 1; display: grid; grid-template-columns: minmax(280px, 38%) 1fr;
         min-height: 0; }
  section { border-right: 1px solid var(--border); display: flex;
            flex-direction: column; min-height: 0; }
  section:last-child { border-right: none; }
  section h2 { font-size: 11px; letter-spacing: 1px; color: var(--dim2);
               text-transform: uppercase; padding: 8px 14px;
               border-bottom: 1px solid var(--border); }
  .scroll { overflow-y: auto; padding: 10px 14px; flex: 1; }
  #log div { white-space: pre-wrap; padding: 1px 0; color: var(--dim2); }
  #log .selection       { color: var(--text); }
  #log .expansion       { color: var(--accent); }
  #log .evaluation      { color: var(--warn); }
  #log .backtracking    { color: var(--bad); font-weight: 600; }
  #log .succeeded       { color: var(--ok); font-weight: 600; }
  #log .exhausted, #log .failed { color: var(--bad); font-weight: 600; }
  #tree ul { list-style: none; padding-left: 18px; border-left: 1px solid var(--border); }
  #tree > ul { padding-left: 0; border-left: none; }
  #tree li { padding: 2px 0; }
  .node-line { display: flex; gap: 8px; align-items: baseline; }
  .glyph { width: 14px; text-align: center; }
  .content { overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
             max-width: 46em; }
  .stats { color: var(--dim2); font-size: 11px; flex-shrink: 0; }
  .st-evaluated .glyph { color: var(--accent); }
  .st-terminal  .glyph { color: var(--ok); }
  .st-terminal  .content { color: var(--ok); }
  .st-pruned    .content, .st-pruned .glyph { color: var(--dim2);
             text-decoration: line-through; }
  .st-failed    .glyph { color: var(--bad); }
  footer { border-top: 1px solid var(--border); max-height: 34%; overflow-y: auto; }
  footer h2 { font-size: 11px; letter-spacing: 1px; color: var(--dim2);
              text-transform: uppercase; padding: 8px 14px; }
  #solution { padding: 0 14px 14px; white-space: pre-wrap; color: var(--ok); }
</style>
</head>
<body>
<header>
  <h1>CognitiveTree-AI</h1>
  <span id="status" class="running">connecting</span>
  <span id="meta" style="color: var(--dim2)"></span>
</header>
<main>
  <section>
    <h2>Phase Log</h2>
    <div id="log" class="scroll"></div>
  </section>
  <section>
    <h2>Thought Tree</h2>
    <div id="tree" class="scroll"></div>
  </section>
</main>
<footer>
  <h2>Solution</h2>
  <pre id="solution">(pending)</pre>
</footer>
<script>
"use strict";
const logEl = document.getElementById("log");
const treeEl = document.getElementById("tree");
const statusEl = document.getElementById("status");
const metaEl = document.getElementById("meta");
const solutionEl = document.getElementById("solution");

const GLYPHS = { pending: "?", evaluated: "*", terminal: "#", pruned: "x", failed: "!" };

function esc(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

function appendLog(env) {
  const line = document.createElement("div");
  line.className = env.phase;
  const node = env.node_id ? " node=" + env.node_id.slice(0, 8) : "";
  const detail = env.detail ? " | " + env.detail : "";
  line.textContent = `[${String(env.iteration).padStart(3, "0")}] ${env.phase}${node}${detail}`;
  logEl.appendChild(line);
  logEl.scrollTop = logEl.scrollHeight;
}

function renderNode(node) {
  const stats = `d${node.depth} v${node.visits} s${node.score.toFixed(2)}`;
  const content = node.content === "" ? "<root>" : node.content.split("\\n")[0];
  let html = `<li class="st-${node.status}"><div class="node-line">` +
    `<span class="glyph">${GLYPHS[node.status] || "?"}</span>` +
    `<span class="content" title="${esc(node.content)}">${esc(content)}</span>` +
    `<span class="stats">${stats}</span></div>`;
  if (node.children.length) {
    html += "<ul>" + node.children.map(renderNode).join("") + "</ul>";
  }
  return html + "</li>";
}

function renderTree(env) {
  treeEl.innerHTML = "<ul>" + renderNode(env.tree.root) + "</ul>";
  metaEl.textContent = `${env.tree.size} nodes`;
}

function finish(env) {
  statusEl.textContent = env.outcome;
  statusEl.className = env.outcome;
  metaEl.textContent += ` | ${env.iterations} iterations`;
  if (env.solution) {
    solutionEl.textContent = env.solution;
  } else {
    solutionEl.textContent = env.error ? "run failed: " + env.error : "(no solution found)";
  }
}

const source = new EventSource("/stream");
source.onopen = () => { statusEl.textContent = "running"; };
source.addEventListener("phase", e => appendLog(JSON.parse(e.data)));
source.addEventListener("snapshot", e => renderTree(JSON.parse(e.data)));
source.addEventListener("result", e => { finish(JSON.parse(e.data)); source.close(); });
source.onerror = () => {
  if (statusEl.className === "running") {
    statusEl.textContent = "disconnected";
    statusEl.className = "failed";
  }
  source.close();
};
</script>
</body>
</html>
"""
