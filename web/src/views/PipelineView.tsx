import { ArrowRight, Check, CheckCircle2, Info } from "lucide-react";
import { useEffect, useState } from "react";
import type { AppConfig, CandidateSummary, StageKey, StageMap } from "../lib/api";
import { COMPONENTS, COMPONENT_COLOR, NEUTRAL, flagLabel, median, topBy } from "../lib/derive";
import { fmtInt, fmtNum, fmtTime } from "../lib/format";
import { STAGE_DOCS } from "../lib/stageDocs";
import { STAGE_ORDER, type RunState } from "../lib/useRun";
import {
  BarList, ComponentLegend, ContributionBar, GroupScatter, Heatmap, Histogram, SlopeChart,
} from "../charts/charts";

export function PipelineView({
  cfg, run, onOpen, onShortlist,
}: { cfg: AppConfig; run: RunState; onOpen: (i: number) => void; onShortlist: () => void }) {
  const [picked, setPicked] = useState<StageKey | null>(null);
  const st = run.stages;
  const lastDone = [...STAGE_ORDER].reverse().find((k) => st[k]);

  // Follow the live run until the presenter clicks a stage.
  useEffect(() => { if (run.status === "running" && !st.ingest) setPicked(null); }, [run.status, st.ingest]);
  const shown: StageKey = picked ?? lastDone ?? "ingest";
  const totalCompute = STAGE_ORDER.reduce((a, k) => a + (st[k]?.seconds ?? 0), 0);
  const cands = st.ingest?.data.candidates ?? [];

  return (
    <>
      <div className="page-head">
        <div>
          <h1>Pipeline</h1>
          <p>{run.meta?.sourceLabel} · {fmtInt(run.n ?? 0)} profiles · {run.meta?.jdLabel}</p>
        </div>
        <div className="actions">
          {run.status === "done" && (
            <button className="btn btn-primary" onClick={onShortlist}>View ranked shortlist <ArrowRight size={15} /></button>
          )}
        </div>
      </div>

      <Kpis run={run} />

      {run.status === "error" && (
        <div className="callout callout-crit mb-20"><Info size={15} /><span><b>Pipeline failed.</b> {run.error}</span></div>
      )}

      <div className="pipe">
        <nav className="card timeline" aria-label="Pipeline stages">
          {cfg.stages.map((s, idx) => {
            const ev = st[s.key];
            const state = ev ? "done" : run.active === s.key && run.status === "running" ? "active" : "pending";
            return (
              <button key={s.key} className={`tl-item ${state}${shown === s.key ? " sel" : ""}`}
                      disabled={!ev && state !== "active"} onClick={() => setPicked(s.key)}>
                <span className="tl-icon">
                  {state === "done" ? <Check size={14} strokeWidth={3} /> : state === "active" ? <span className="spinner" /> : idx + 1}
                </span>
                <span>
                  <span className="tl-title" style={{ display: "block" }}>{s.title}</span>
                  <span className="tl-desc">{s.description}</span>
                </span>
                <span className="tl-time">{ev ? fmtTime(ev.seconds) : state === "active" ? "Running" : ""}</span>
              </button>
            );
          })}
          <div className="tl-total">
            <span>Compute time</span>
            <b>{fmtTime(totalCompute)}</b>
          </div>
          {picked && run.status === "running" && (
            <button className="btn btn-ghost" style={{ width: "100%", marginTop: 4 }} onClick={() => setPicked(null)}>Follow live</button>
          )}
        </nav>

        <StagePanel cfg={cfg} stageKey={shown} st={st} cands={cands} onOpen={onOpen} />
      </div>
    </>
  );
}

function Kpis({ run }: { run: RunState }) {
  const st = run.stages;
  const nHp = st.trust ? st.trust.data.hp.filter((v) => v < 1).length : null;
  const nDq = st.trust ? st.trust.data.dq.filter((v) => v < 1).length : null;
  const item = (label: string, v: number | null | undefined, note: string, cls = "") => (
    <div className={`kpi${cls ? ` kpi-${cls}` : ""}`}>
      <div className="kpi-label">{label}</div>
      <div className="kpi-value">{v === null || v === undefined ? <span className="muted">—</span> : fmtInt(v)}</div>
      <div className="kpi-note">{note}</div>
    </div>
  );
  return (
    <div className="card kpis">
      {item("Screened", run.n, "Profiles in the pool")}
      {item("Likely fake or inconsistent", nHp, "Honeypot checks", nHp ? "crit" : "")}
      {item("JD disqualifier", nDq, "Penalised, not removed", nDq ? "warn" : "")}
      {item("Shortlisted", st.stage1?.data.k, "Sent to the cross-encoder")}
      {item("Ranked & explained", st.final?.data.rows.length, "Final output", st.final ? "brand" : "")}
    </div>
  );
}

function StagePanel({
  cfg, stageKey, st, cands, onOpen,
}: { cfg: AppConfig; stageKey: StageKey; st: StageMap; cands: CandidateSummary[]; onOpen: (i: number) => void }) {
  const idx = STAGE_ORDER.indexOf(stageKey);
  const def = cfg.stages[idx];
  const ev = st[stageKey];
  const doc = STAGE_DOCS[stageKey];
  return (
    <section className="card" aria-live="polite">
      <div className="stage-head">
        <div>
          <div className="stage-kicker">Stage {idx + 1} of {STAGE_ORDER.length}</div>
          <h2 className="stage-title">{def.title} <span className="muted" style={{ fontWeight: 400, fontSize: 15 }}>· {def.description}</span></h2>
        </div>
        {ev && <div className="stage-time"><div className="v">{fmtTime(ev.seconds)}</div><div className="l">compute time</div></div>}
      </div>
      <div className="stage-explain">
        <div><b>What happens</b>{doc.what}</div>
        <div><b>Why it matters</b>{doc.why}</div>
      </div>
      {ev ? (
        <>
          <div className="stage-result"><CheckCircle2 size={15} /><span>{ev.headline}</span></div>
          <div className="stage-body"><StageBody cfg={cfg} k={stageKey} st={st} cands={cands} onOpen={onOpen} /></div>
        </>
      ) : (
        <div className="stage-wait"><span className="spinner" style={{ width: 22, height: 22 }} />Running {def.title.toLowerCase()}…</div>
      )}
    </section>
  );
}

function TopTable({
  idxs, values, cands, onOpen, label,
}: { idxs: number[]; values: number[]; cands: CandidateSummary[]; onOpen: (i: number) => void; label: string }) {
  return (
    <div className="table-wrap card" style={{ boxShadow: "none" }}>
      <table className="tbl compact">
        <thead><tr><th>#</th><th>Candidate</th><th className="r">{label}</th></tr></thead>
        <tbody>
          {idxs.map((i, r) => (
            <tr key={i} className="click" onClick={() => onOpen(i)}>
              <td className="num muted">{r + 1}</td>
              <td><div className="cell-main">{cands[i]?.title || "—"}</div><div className="cell-sub">{cands[i]?.company} · <span className="mono">{cands[i]?.id}</span></div></td>
              <td className="r">{fmtNum(values[i], 3)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function StageBody({
  cfg, k, st, cands, onOpen,
}: { cfg: AppConfig; k: StageKey; st: StageMap; cands: CandidateSummary[]; onOpen: (i: number) => void }) {
  switch (k) {
    case "ingest": {
      const d = st.ingest!.data;
      return (
        <>
          <div className="stat-row">
            <div className="stat"><div className="l">Profiles parsed</div><div className="v">{fmtInt(d.candidates.length)}</div></div>
            <div className="stat"><div className="l">Duplicate IDs</div><div className="v">{d.dupes}</div></div>
            <div className="stat"><div className="l">Median experience</div><div className="v">{median(d.yoe).toFixed(1)} yrs</div></div>
          </div>
          <div className="stage-cols">
            <div><div className="panel-title">Years of experience <span className="hint">JD target: 6–8 years</span></div>
              <Histogram values={d.yoe} xLabel="Years of experience" digits={1} /></div>
            <div><div className="panel-title">Completeness</div>
              <BarList items={Object.entries(d.missing).map(([f, v]) => ({ label: `Missing ${f.replace(/_/g, " ")}`, value: v }))}
                       max={Math.max(d.candidates.length, 1)} fmt={(v) => fmtInt(v)} color="#9aa4b2" />
              <p className="muted mt-12" style={{ fontSize: 12.5 }}>Profiles missing a section are still scored; the affected signals contribute zero.</p></div>
          </div>
        </>
      );
    }
    case "text": {
      const d = st.text!.data;
      return (
        <>
          <div className="stat-row">
            <div className="stat"><div className="l">Avg. representation length</div><div className="v">{fmtInt(Math.round(d.avg_chars))} chars</div></div>
            <div className="stat"><div className="l">Avg. keyword tokens</div><div className="v">{fmtInt(Math.round(d.avg_tokens))}</div></div>
            <div className="stat"><div className="l">Unique JD query terms</div><div className="v">{d.jd_terms}</div></div>
          </div>
          <div className="grid-2">
            <div><div className="panel-title">Job description as the models read it</div><div className="textbox">{d.jd_text}</div></div>
            <div><div className="panel-title">Example profile representation <span className="hint">{cands[0]?.id}</span></div><div className="textbox">{d.example_text}</div></div>
          </div>
        </>
      );
    }
    case "semantic": {
      const d = st.semantic!.data;
      return (
        <div className="stage-cols">
          <div><div className="panel-title">Similarity distribution <span className="hint">{d.dim}-dimensional embeddings</span></div>
            <Histogram values={d.sem} xLabel="Cosine similarity to the job description" color={COMPONENT_COLOR.Semantic} /></div>
          <div><div className="panel-title">Top 5 by semantic score alone</div>
            <TopTable idxs={topBy(d.sem, 5)} values={d.sem} cands={cands} onOpen={onOpen} label="Similarity" /></div>
        </div>
      );
    }
    case "lexical": {
      const d = st.lexical!.data;
      return (
        <div className="stage-cols">
          <div><div className="panel-title">Most discriminative JD terms in this pool <span className="hint">inverse document frequency</span></div>
            <BarList items={d.terms.slice(0, 12).map((t) => ({ label: t.term, value: t.idf }))} color={COMPONENT_COLOR.Lexical} /></div>
          <div><div className="panel-title">Top 5 by keyword score alone</div>
            <TopTable idxs={topBy(d.lex, 5)} values={d.lex} cands={cands} onOpen={onOpen} label="BM25" /></div>
        </div>
      );
    }
    case "structured": {
      const d = st.structured!.data;
      const keys = Object.keys(cfg.weights.structured);
      const top = topBy(d.fit, 10);
      return (
        <>
          <div className="panel-title">Sub-scores for the 10 strongest JD-fit profiles</div>
          <div className="table-wrap" style={{ marginBottom: 22 }}>
            <Heatmap cols={keys.map((k) => ({ label: k.charAt(0).toUpperCase() + k.slice(1), sub: `${Math.round(cfg.weights.structured[k] * 100)}%` }))}
                     rows={top.map((i) => ({ label: `${cands[i]?.title || cands[i]?.id} · ${cands[i]?.company}`, values: keys.map((k) => d.subs[i][k] ?? 0) }))} />
          </div>
          <div className="grid-2">
            <div><div className="panel-title">Signal weights</div>
              <BarList items={keys.map((k) => ({ label: k.charAt(0).toUpperCase() + k.slice(1), value: cfg.weights.structured[k] }))}
                       fmt={(v) => `${Math.round(v * 100)}%`} color={COMPONENT_COLOR["JD-fit"]} /></div>
            <div><div className="panel-title">JD-fit distribution</div>
              <Histogram values={d.fit} xLabel="Composite JD-fit score" color={COMPONENT_COLOR["JD-fit"]} height={160} /></div>
          </div>
        </>
      );
    }
    case "trust": {
      const d = st.trust!.data;
      const counts = new Map<string, { n: number; kind: "hp" | "dq" }>();
      d.hp_flags.forEach((fl) => fl.forEach((f) => { const l = flagLabel(f); counts.set(l, { n: (counts.get(l)?.n ?? 0) + 1, kind: "hp" }); }));
      d.dq_flags.forEach((fl) => fl.forEach((f) => { const l = flagLabel(f); counts.set(l, { n: (counts.get(l)?.n ?? 0) + 1, kind: "dq" }); }));
      const items = [...counts.entries()].sort((a, b) => b[1].n - a[1].n)
        .map(([label, v]) => ({ label, value: v.n, color: v.kind === "hp" ? "#a8433d" : "#c08a2e" }));
      const nHp = d.hp.filter((v) => v < 1).length, nDq = d.dq.filter((v) => v < 1).length;
      const clean = d.hp.filter((v, i) => v >= 1 && d.dq[i] >= 1).length;
      const worst = d.hp.map((v, i) => ({ i, m: v * d.dq[i] })).filter((x) => x.m < 1).sort((a, b) => a.m - b.m).slice(0, 6);
      return (
        <>
          <div className="stat-row">
            <div className="stat"><div className="l">Honeypot flags</div><div className="v" style={{ color: "var(--crit)" }}>{fmtInt(nHp)}</div></div>
            <div className="stat"><div className="l">Disqualifier hits</div><div className="v" style={{ color: "var(--warn)" }}>{fmtInt(nDq)}</div></div>
            <div className="stat"><div className="l">Clean profiles</div><div className="v">{fmtInt(clean)}</div></div>
          </div>
          <div className="stage-cols">
            <div>
              <div className="panel-title">Triggered checks</div>
              <div className="legend" style={{ marginBottom: 10 }}>
                <span><i style={{ background: "#a8433d" }} />Honeypot (inconsistent profile)</span>
                <span><i style={{ background: "#c08a2e" }} />JD disqualifier</span>
              </div>
              {items.length ? <BarList items={items} fmt={(v) => fmtInt(v)} /> : <p className="muted">No checks triggered.</p>}
            </div>
            <div>
              <div className="panel-title">Most heavily penalised</div>
              <div className="table-wrap card" style={{ boxShadow: "none" }}>
                <table className="tbl compact">
                  <thead><tr><th>Candidate</th><th className="r">Multiplier</th></tr></thead>
                  <tbody>{worst.map(({ i, m }) => (
                    <tr key={i} className="click" onClick={() => onOpen(i)}>
                      <td><div className="cell-main">{cands[i]?.title || cands[i]?.id}</div>
                        <div className="cell-sub">{[...d.hp_flags[i], ...d.dq_flags[i]].map(flagLabel).join(" · ")}</div></td>
                      <td className="r">×{m.toFixed(2)}</td>
                    </tr>))}</tbody>
                </table>
              </div>
            </div>
          </div>
        </>
      );
    }
    case "stage1": {
      const d = st.stage1!.data, sem = st.semantic!.data.sem, fit = st.structured!.data.fit, t = st.trust!.data;
      const top = new Set(d.top);
      const points = sem.map((x, i) => ({
        x, y: fit[i], label: cands[i]?.id ?? String(i), sub: `${cands[i]?.title} · ${cands[i]?.company}`,
        group: top.has(i) ? "short" : t.hp[i] * t.dq[i] < 1 ? "pen" : "rest",
      }));
      return (
        <>
          <p className="muted" style={{ fontSize: 13, marginBottom: 12 }}>
            Each point is one profile. Shortlisted profiles cluster where semantic and JD-fit scores are both high;
            penalised profiles can score well on text yet be held back by the trust stage.
          </p>
          <GroupScatter points={points} xLabel="Semantic similarity" yLabel="JD-fit score"
                        groups={[{ key: "short", label: "Shortlisted", color: "#2f6297" },
                                 { key: "pen", label: "Penalised", color: "#c47c3c" },
                                 { key: "rest", label: "Not shortlisted", color: NEUTRAL }]} />
        </>
      );
    }
    case "rerank": {
      const d = st.rerank!.data;
      if (!d.enabled) return <div className="callout callout-info"><Info size={15} /><span>Cross-encoder reranking was turned off for this run. The stage-1 order carries through and the cross-encoder term contributes zero.</span></div>;
      const top = st.stage1!.data.top;
      const points = top.map((ci, p) => ({ x: p + 1, y: d.ce[p], label: cands[ci]?.id ?? "", sub: `${cands[ci]?.title} · ${cands[ci]?.company}`, group: "ce" }));
      return (
        <div className="stage-cols">
          <div><div className="panel-title">Cross-encoder relevance by stage-1 position</div>
            <GroupScatter points={points} groups={[{ key: "ce", label: "Shortlisted profile", color: COMPONENT_COLOR["Cross-encoder"] }]}
                          xLabel="Stage-1 rank" yLabel="Cross-encoder relevance" xFmt={(v) => v.toFixed(0)} yDomain={[0, 1]} height={300} /></div>
          <div><div className="panel-title">Highest cross-encoder relevance</div>
            <TopTable idxs={topBy(d.ce, 5).map((p) => top[p])} values={(() => { const m: number[] = []; top.forEach((ci, p) => { m[ci] = d.ce[p]; }); return m; })()}
                      cands={cands} onOpen={onOpen} label="Relevance" /></div>
        </div>
      );
    }
    case "final": {
      const rows = st.final!.data.rows;
      const ce = st.rerank?.data.enabled;
      return (
        <div className="stage-cols">
          <div>
            <div className="panel-title">{ce ? "Rank movement after reranking" : "Final order"}</div>
            <SlopeChart items={rows.map((r) => ({ id: r.id, label: cands[r.i]?.title || r.id, from: r.s1_rank, to: r.rank }))} />
          </div>
          <div>
            <div className="panel-title">Score build-up, top 10</div>
            <div style={{ marginBottom: 12 }}><ComponentLegend weights={cfg.weights.stage2} /></div>
            <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
              {rows.slice(0, 10).map((r) => (
                <button key={r.i} className="btn-ghost" onClick={() => onOpen(r.i)}
                        style={{ border: 0, background: "transparent", padding: 0, textAlign: "left", cursor: "pointer" }}>
                  <div className="row-gap" style={{ fontSize: 12.5, marginBottom: 4 }}>
                    <b className="num" style={{ color: "var(--brand)", width: 26 }}>#{r.rank}</b>
                    <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{cands[r.i]?.title} · {cands[r.i]?.company}</span>
                    <span className="num" style={{ marginLeft: "auto", fontWeight: 600 }}>{r.score.toFixed(3)}</span>
                  </div>
                  <ContributionBar contrib={r.contrib} />
                </button>
              ))}
            </div>
            <p className="muted mt-12" style={{ fontSize: 12 }}>
              Bars show each component's weighted contribution out of a perfect 1.0, before penalties. Displayed scores are
              rescaled to 0.20–0.99 within the list. {COMPONENTS.length} components, weights fixed in config.
            </p>
          </div>
        </div>
      );
    }
  }
}
