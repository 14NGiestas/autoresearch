#!/usr/bin/env node
// rechain_registry.mjs — repara e migra a registry do HEP para o protocolo JS.
//
// Por que existe: a registry foi escrita pelo hep.py (Python, aposentado). Duas
// coisas a deixaram inauditavel para a implementacao de referencia:
//   1) um seq DUPLICADO (96) — dois processos leram o mesmo tail e ambos
//      anexaram, entao a cadeia tem dois elos com o mesmo numero;
//   2) valores de `kind` fora do enum do protocolo (test/empirical/measurement/
//      note/external/"supports"), que a validacao do JS rejeita.
//
// O que este script faz, e SO isso:
//   * mapeia `kind` invalido para o enum (test|empirical|measurement ->
//     experiment, external -> literature, note|refine -> analysis,
//     supports|inconclusive -> experiment);
//   * renumera `seq` na ordem do arquivo (a ordem real de append e a verdade);
//   * recalcula `prev` e `hash` com a canonicalizacao da referencia.
// Nenhum payload e removido, reordenado ou reescrito alem do `kind`.
//
// Uso: node scripts/rechain_registry.mjs [--registry hep/registry.jsonl] [--dry]
import { createHash } from "node:crypto";
import { readFileSync, writeFileSync, copyFileSync } from "node:fs";

function stableStringify(v) {
  const raw = (() => {
    if (v === null || typeof v !== "object") return JSON.stringify(v);
    if (Array.isArray(v)) return `[${v.map(stableStringify).join(", ")}]`;
    const keys = Object.keys(v).sort();
    return `{${keys.map((k) => `${JSON.stringify(k)}: ${stableStringify(v[k])}`).join(", ")}}`;
  })();
  // mesma regra da referencia: escapa nao-ASCII como \uXXXX (compat. json.dumps)
  return raw.replace(/[\u0080-\uFFFF]/g, (c) => `\\u${c.charCodeAt(0).toString(16).padStart(4, "0")}`);
}
const sha = (s) => createHash("sha256").update(s).digest("hex");

const KIND_MAP = {
  test: "experiment", empirical: "experiment", measurement: "experiment",
  supports: "experiment", inconclusive: "experiment",
  external: "literature", note: "analysis", refine: "analysis",
};
const VALID = new Set(["simulation", "experiment", "literature", "derivation", "analysis"]);

const argv = process.argv.slice(2);
const get = (f) => { const i = argv.indexOf(f); return i >= 0 ? argv[i + 1] : undefined; };
const path = get("--registry") ?? "hep/registry.jsonl";
const dry = argv.includes("--dry");

const events = readFileSync(path, "utf8").split("\n").filter((l) => l.trim()).map((l) => JSON.parse(l));
let prev = "0".repeat(64);
let kindsFixed = 0, hashesChanged = 0;
const out = [];
events.forEach((e, i) => {
  const seq = i + 1;
  const kind = e.payload?.kind;
  if (kind && !VALID.has(kind) && KIND_MAP[kind]) { e.payload.kind = KIND_MAP[kind]; kindsFixed++; }
  const hash = sha(`${prev}${stableStringify(e.payload)}`);
  if (hash !== e.hash || e.seq !== seq || e.prev !== prev) hashesChanged++;
  out.push({ seq, prev, type: e.type, ts: e.ts, payload: e.payload, hash });
  prev = hash;
});
console.log(`registros: ${out.length}`);
console.log(`kind migrados: ${kindsFixed}`);
console.log(`elos reescritos (seq/prev/hash): ${hashesChanged}`);
const stillBad = out.filter((e) => e.payload?.kind && !VALID.has(e.payload.kind));
console.log(`kinds ainda fora do enum: ${stillBad.length}`);
if (dry) { console.log("(dry run, nada escrito)"); process.exit(0); }
copyFileSync(path, `${path}.pre-rechain`);
writeFileSync(path, out.map((e) => JSON.stringify(e)).join("\n") + "\n");
console.log(`escrito ${path} (backup em ${path}.pre-rechain)`);
