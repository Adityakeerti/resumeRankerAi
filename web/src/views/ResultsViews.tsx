import { Download, FileSpreadsheet, FileText, Search } from "lucide-react";
import { useMemo, useState } from "react";
import type { AppConfig } from "../lib/api";
import { flagLabel, type Results } from "../lib/derive";
import { csvCell, downloadText, fmtInt, fmtNum } from "../lib/format";
import type { RunState } from "../lib/useRun";
import { BarList, ComponentLegend, ContributionBar } from "../charts/charts";
import { Card, Delta, Segmented } from "../components/ui";

/* ── Shortlist ─────────────────────────────────────────────────────────── */
export function ShortlistView({
  cfg, res, onOpen, selected,
}: { cfg: AppConfig; res: Results; onOpen: (i: number) => void; selected: number | null }) {
  const [q, setQ] = useState("");
  const rows = useMemo(() => {
    const t = q.trim().toLowerCase();
    if (!t) return res.rows;
    return res.rows.filter((r) => {
      const c = res.cands[r.i];
      return [c.id, c.name, c.title, c.company, c.location].some((s) => s.toLowerCase().includes(t));
    });
  }, [q, res]);

  return (
    <>
      <div className="page-head">
        <div>
          <h1>Ranked shortlist</h1>
          <p>{res.rows.length} of {fmtInt(res.n)} profiles, ordered by fused score. Select a candidate for the full evidence trail.</p>
        </div>
      </div>
      <Card title="Candidates" sub={`${rows.length} shown`} bodyClass=""
            right={
              <div className="row-gap" style={{ position: "relative" }}>
                <Search size={14} style={{ position: "absolute", left: 10, color: "var(--ink-3)" }} />
                <input className="input" style={{ width: 240, paddingLeft: 30, height: 32 }} placeholder="Filter by name, role, company"
                       value={q} onChange={(e) => setQ(e.target.value)} aria-label="Filter candidates" />
              </div>
            }>
        <div style={{ padding: "12px 20px", borderBottom: "1px solid var(--border)" }}>
          <ComponentLegend weights={cfg.weights.stage2} />
        </div>
        <div className="table-wrap">
          <table className="tbl">
            <thead>
              <tr>
                <th>Rank</th><th>Candidate</th><th>Current role</th><th>Location</th>
                <th className="r">Exp.</th><th className="r">Notice</th><th style={{ width: 180 }}>Score build-up</th>
                <th className="r">Score</th><th className="r" title="Change from stage-1 rank after reranking">Rerank</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => {
                const c = res.cands[r.i];
                const flagged = res.hp[r.i] < 1 || res.dq[r.i] < 1;
                return (
                  <tr key={r.i} className={`click${selected === r.i ? " sel" : ""}`} onClick={() => onOpen(r.i)}>
                    <td className="rank-cell">{r.rank}</td>
                    <td>
                      <div className="cell-main">{c.name || c.id}</div>
                      <div className="cell-sub mono">{c.id}</div>
                    </td>
                    <td>
                      <div className="cell-main" style={{ fontWeight: 500 }}>{c.title || "—"}</div>
                      <div className="cell-sub">{c.company || "—"}{flagged && <span className="badge badge-warn" style={{ marginLeft: 8, height: 18 }}>Penalised</span>}</div>
                    </td>
                    <td className="cell-sub" style={{ color: "var(--ink-2)" }}>{c.location || "—"}</td>
                    <td className="r">{c.yoe.toFixed(1)}y</td>
                    <td className="r">{c.notice ?? "—"}{c.notice !== null && "d"}</td>
                    <td><ContributionBar contrib={r.contrib} /></td>
                    <td className="r score-pill">{r.score.toFixed(3)}</td>
                    <td className="r"><Delta from={r.s1_rank} to={r.rank} enabled={res.ceEnabled} /></td>
                  </tr>
                );
              })}
              {!rows.length && <tr><td colSpan={9}><div className="empty">No candidates match “{q}”.</div></td></tr>}
            </tbody>
          </table>
        </div>
      </Card>
      <p className="muted mt-12" style={{ fontSize: 12 }}>
        Scores are min–max rescaled to 0.20–0.99 within this list: they order candidates and are not probabilities.
      </p>
    </>
  );
}

/* ── Risk review ───────────────────────────────────────────────────────── */
type RiskFilter = "all" | "hp" | "dq";
export function RiskView({ res, onOpen }: { res: Results; onOpen: (i: number) => void }) {
  const [f, setF] = useState<RiskFilter>("all");
  const flagged = useMemo(() => res.cands.map((c) => c.i).filter((i) => res.hp[i] < 1 || res.dq[i] < 1), [res]);
  const list = flagged
    .filter((i) => (f === "hp" ? res.hp[i] < 1 : f === "dq" ? res.dq[i] < 1 : true))
    .sort((a, b) => res.hp[a] * res.dq[a] - res.hp[b] * res.dq[b]);
  const reasons = new Map<string, { n: number; kind: "hp" | "dq" }>();
  res.hpFlags.forEach((fl) => fl.forEach((x) => { const l = flagLabel(x); reasons.set(l, { n: (reasons.get(l)?.n ?? 0) + 1, kind: "hp" }); }));
  res.dqFlags.forEach((fl) => fl.forEach((x) => { const l = flagLabel(x); reasons.set(l, { n: (reasons.get(l)?.n ?? 0) + 1, kind: "dq" }); }));
  const stillRanked = flagged.filter((i) => res.finalRank.has(i)).length;
  const nHp = res.hp.filter((v) => v < 1).length, nDq = res.dq.filter((v) => v < 1).length;

  return (
    <>
      <div className="page-head">
        <div>
          <h1>Risk review</h1>
          <p>Profiles that triggered a consistency check or a job-description exclusion. Penalties lower a score; they never delete a profile outright.</p>
        </div>
      </div>
      <div className="card kpis" style={{ gridTemplateColumns: "repeat(4, minmax(0,1fr))" }}>
        <div className="kpi"><div className="kpi-label">Flagged in total</div><div className="kpi-value">{fmtInt(flagged.length)}</div><div className="kpi-note">of {fmtInt(res.n)} screened</div></div>
        <div className="kpi kpi-crit"><div className="kpi-label">Likely fake or inconsistent</div><div className="kpi-value">{fmtInt(nHp)}</div><div className="kpi-note">Honeypot checks</div></div>
        <div className="kpi kpi-warn"><div className="kpi-label">JD disqualifier</div><div className="kpi-value">{fmtInt(nDq)}</div><div className="kpi-note">Explicit exclusions</div></div>
        <div className="kpi"><div className="kpi-label">Flagged but still ranked</div><div className="kpi-value">{fmtInt(stillRanked)}</div><div className="kpi-note">Strong enough to survive a penalty</div></div>
      </div>

      <div className="grid-2" style={{ gridTemplateColumns: "minmax(0,0.85fr) minmax(0,1.6fr)", alignItems: "start" }}>
        <Card title="By reason">
          <div className="legend" style={{ marginBottom: 12 }}>
            <span><i style={{ background: "#a8433d" }} />Honeypot</span>
            <span><i style={{ background: "#c08a2e" }} />Disqualifier</span>
          </div>
          {reasons.size ? (
            <BarList items={[...reasons.entries()].sort((a, b) => b[1].n - a[1].n)
              .map(([label, v]) => ({ label, value: v.n, color: v.kind === "hp" ? "#a8433d" : "#c08a2e" }))} fmt={(v) => fmtInt(v)} />
          ) : <p className="muted">No checks were triggered in this pool.</p>}
        </Card>
        <Card title="Flagged profiles" sub={`${list.length} shown`} bodyClass=""
              right={<Segmented<RiskFilter> ariaLabel="Filter" value={f} onChange={setF}
                                options={[{ value: "all", label: "All" }, { value: "hp", label: "Honeypot" }, { value: "dq", label: "Disqualifier" }]} />}>
          <div className="table-wrap" style={{ maxHeight: 620, overflowY: "auto" }}>
            <table className="tbl">
              <thead><tr><th>Candidate</th><th>Findings</th><th className="r">Multiplier</th><th className="r">Outcome</th></tr></thead>
              <tbody>
                {list.map((i) => {
                  const c = res.cands[i];
                  return (
                    <tr key={i} className="click" onClick={() => onOpen(i)}>
                      <td style={{ minWidth: 180 }}><div className="cell-main">{c.title || "—"}</div><div className="cell-sub">{c.company} · <span className="mono">{c.id}</span></div></td>
                      <td><div className="tags">
                        {res.hpFlags[i].map((x) => <span key={x} className="badge badge-crit" title={x}>{flagLabel(x)}</span>)}
                        {res.dqFlags[i].map((x) => <span key={x} className="badge badge-warn" title={x}>{flagLabel(x)}</span>)}
                      </div></td>
                      <td className="r">×{(res.hp[i] * res.dq[i]).toFixed(2)}</td>
                      <td className="r">{res.finalRank.has(i) ? <span className="badge badge-brand">Ranked #{res.finalRank.get(i)}</span>
                                                                 : <span className="muted" style={{ fontSize: 12, whiteSpace: "nowrap" }}>Stage-1 #{res.s1Rank[i]}</span>}</td>
                    </tr>
                  );
                })}
                {!list.length && <tr><td colSpan={4}><div className="empty">Nothing in this category.</div></td></tr>}
              </tbody>
            </table>
          </div>
        </Card>
      </div>
    </>
  );
}

/* ── Export ────────────────────────────────────────────────────────────── */
export function ExportView({ cfg, res, run }: { cfg: AppConfig; res: Results; run: RunState }) {
  const submission = () => [
    "candidate_id,rank,score,reasoning",
    ...res.rows.map((r) => [csvCell(r.id), csvCell(r.rank), csvCell(r.score.toFixed(4)), csvCell(r.reasoning)].join(",")),
  ].join("\n");
  const trace = () => [
    "candidate_id,semantic,lexical,jd_fit,honeypot_mult,disq_mult,stage1_score,stage1_rank,cross_encoder,final_rank,flags",
    ...res.cands.map((c) => [
      csvCell(c.id), res.sem[c.i], res.lex[c.i], res.fit[c.i], res.hp[c.i], res.dq[c.i], res.s1[c.i], res.s1Rank[c.i],
      res.ce.get(c.i) ?? "", res.finalRank.get(c.i) ?? "", csvCell([...res.hpFlags[c.i], ...res.dqFlags[c.i]].join(" | ")),
    ].join(",")),
  ].join("\n");
  const req = run.request!;
  return (
    <>
      <div className="page-head">
        <div><h1>Export</h1><p>Download the ranked submission or the full per-profile score trace for offline analysis.</p></div>
      </div>
      <div className="export-grid">
        <div className="card export-card">
          <div className="export-icon"><FileText size={19} /></div>
          <div style={{ flex: 1 }}>
            <h3>submission.csv</h3>
            <p>{res.rows.length} rows · candidate_id, rank, score, reasoning. Challenge submission format.</p>
            <button className="btn btn-primary" onClick={() => downloadText("submission.csv", submission())}><Download size={15} /> Download</button>
          </div>
        </div>
        <div className="card export-card">
          <div className="export-icon"><FileSpreadsheet size={19} /></div>
          <div style={{ flex: 1 }}>
            <h3>pipeline_trace.csv</h3>
            <p>{fmtInt(res.n)} rows · every component score, multiplier and rank for every profile in the pool.</p>
            <button className="btn" onClick={() => downloadText("pipeline_trace.csv", trace())}><Download size={15} /> Download</button>
          </div>
        </div>
      </div>
      <div className="grid-2" style={{ alignItems: "start" }}>
        <Card title="Preview" sub="submission.csv, first 8 rows" bodyClass="">
          <div className="table-wrap">
            <table className="tbl compact">
              <thead><tr><th>candidate_id</th><th className="r">rank</th><th className="r">score</th><th>reasoning</th></tr></thead>
              <tbody>{res.rows.slice(0, 8).map((r) => (
                <tr key={r.i}><td className="mono">{r.id}</td><td className="r">{r.rank}</td><td className="r">{r.score.toFixed(4)}</td>
                  <td className="cell-sub" style={{ maxWidth: 320, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }} title={r.reasoning}>{r.reasoning}</td></tr>
              ))}</tbody>
            </table>
          </div>
        </Card>
        <Card title="Run configuration">
          <dl className="cfg">
            <dt>Run ID</dt><dd className="mono">{run.runId}</dd>
            <dt>Talent pool</dt><dd>{run.meta?.sourceLabel} · {fmtInt(res.n)} profiles{req.source === "full" && ` · seed ${req.seed}`}</dd>
            <dt>Job description</dt><dd>{run.meta?.jdLabel} · {fmtInt(req.jd_text.length)} characters</dd>
            <dt>Shortlist / final</dt><dd>{req.shortlist_k} / {req.final_n}</dd>
            <dt>Cross-encoder</dt><dd>{req.use_cross_encoder ? cfg.models.cross_encoder : "Off"}</dd>
            <dt>Bi-encoder</dt><dd>{cfg.models.bi_encoder}</dd>
            <dt>Stage-2 weights</dt><dd className="num">{Object.entries(cfg.weights.stage2).map(([k, v]) => `${k} ${fmtNum(v)}`).join(" · ")}</dd>
            <dt>Reference date</dt><dd>{run.today}</dd>
            <dt>Wall time</dt><dd className="num">{fmtNum(run.wall)} s</dd>
          </dl>
        </Card>
      </div>
    </>
  );
}
