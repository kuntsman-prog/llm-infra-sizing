// Запускает расчётное ядро веб-версии (docs/index.html) на сценарии из stdin и печатает результат в JSON.
// Используется тестом tests/test_web_parity.py, чтобы веб-версия и Python давали одинаковые цифры.
const fs = require("fs");
const path = require("path");
const html = fs.readFileSync(path.join(__dirname, "..", "docs", "index.html"), "utf8");
const engine = html.split("// ENGINE START")[1].split("// ENGINE END")[0];
const run = new Function(engine + "\nreturn { MODELS, GPUS, ECON, compute };")();
const s = JSON.parse(fs.readFileSync(0, "utf8"));
const econ = { ...run.ECON, apiIn: s.apiIn ?? run.ECON.apiIn, apiOut: s.apiOut ?? run.ECON.apiOut };
const r = run.compute(s, run.MODELS, run.GPUS, econ);
const out = {};
for (const { name, p, c } of r.rows) {
  out[name] = p.feasible
    ? { tp: p.tp, batch: p.batch, replicas: p.replicas, gpus: p.totalGpus, ttft: p.ttft, tpot_ms: p.tpotMs, own: c.own, cloud: c.cloud, api: c.api }
    : null;
}
process.stdout.write(JSON.stringify(out));
