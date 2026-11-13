import { AlertOctagon, AlertTriangle, ChevronRight, X } from "lucide-react";
import { useEffect, useState } from "react";
import { api, type AppConfig, type CandidateDetail } from "../lib/api";
import { COMPONENTS, COMPONENT_COLOR, COMPONENT_WEIGHT_KEY, flagDetail, flagLabel, type Results } from "../lib/derive";
import { fmtInt, fmtNum, fmtPct, titleCase } from "../lib/format";
import { BarList } from "../charts/charts";

export function CandidateDrawer({
  cfg, runId, idx, res, onClose,
}: { cfg: AppConfig; runId: string; idx: number; res: Results; onClose: () => void }) {
  const [detail, setDetail] = useState<CandidateDetail | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    const ctrl = new AbortController();
    setDetail(null); setErr(null);
    api.candidate(runId, idx, ctrl.signal).then(setDetail).catch((e) => { if (!ctrl.signal.aborted) setErr(String(e.message ?? e)); });
    return () => ctrl.abort();
  }, [runId, idx]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const c = res.cands[idx];
  const row = res.rows.find((r) => r.i === idx);
  const ce = res.ce.get(idx);
  const w = cfg.weights.stage2;
  const values: Record<string, number | undefined> = {
    "Semantic": res.sem[idx], "Lexical": res.lex[idx], "JD-fit": res.fit[idx], "Cross-encoder": ce,
  };
  const subs = res.subs[idx];
  const notRanked: string[] = [];
  if (!row) {
    if (res.hp[idx] < 1) notRanked.push("flagged by honeypot checks");
    if (res.dq[idx] < 1) notRanked.push("penalised by a job-description disqualifier");
    notRanked.push(res.shortlist.has(idx)
      ? "fell below the cut after cross-encoder reranking"
      : `stage-1 rank ${res.s1Rank[idx]} is outside the top-${res.shortlist.size} shortlist`);
  }

  return (
    <>
      <div className="scrim" onClick={onClose} />
      <aside className="drawer" role="dialog" aria-modal="true" aria-label={`Candidate ${c.id}`}>
        <header className="drawer-head">
          <div style={{ flex: 1, minWidth: 0 }}>
            <div className="row-gap" style={{ marginBottom: 4 }}>
              <span className="mono muted">{c.id}</span>
              {c.open && <span className="badge badge-ok">Open to work</span>}
            </div>
            <h2>{c.name || c.id}</h2>
            <div style={{ color: "var(--ink-2)" }}>{c.title || "—"} · {c.company || "—"}</div>
            <div className="muted" style={{ fontSize: 12.5 }}>{c.location || "—"} · {c.yoe.toFixed(1)} years · {c.notice ?? "—"}-day notice</div>
          </div>
          <button className="btn btn-ghost btn-icon" onClick={onClose} aria-label="Close"><X size={18} /></button>
        </header>

        <div className="drawer-body">
          {row ? (
            <div className="verdict">
              <div className="big">#{row.rank}</div>
              <div>
                <div style={{ fontWeight: 600 }}>Ranked {row.rank} of {fmtInt(res.n)}</div>
                <div className="muted" style={{ fontSize: 12.5 }}>
                  Score {row.score.toFixed(3)} · stage-1 rank {row.s1_rank}{res.ceEnabled && row.s1_rank !== row.rank && ` → moved ${row.s1_rank > row.rank ? "up" : "down"} ${Math.abs(row.s1_rank - row.rank)} after reranking`}
                </div>
              </div>
            </div>
          ) : (
            <div className="verdict">
              <div className="big" style={{ color: "var(--ink-3)", fontSize: 15, minWidth: 0 }}>Not ranked</div>
              <div className="muted" style={{ fontSize: 13 }}>{titleCase(notRanked.join("; "))}.</div>
            </div>
          )}

          {row && <div className="reason">{row.reasoning}</div>}

          <section className="drawer-sec">
            <h3>Score components</h3>
            <table className="tbl compact" style={{ border: "1px solid var(--border)", borderRadius: 10 }}>
              <thead><tr><th>Component</th><th className="r">Raw</th><th className="r">Weight</th><th className="r">Contribution</th><th style={{ width: 130 }} /></tr></thead>
              <tbody>
                {COMPONENTS.map((k) => {
                  const v = values[k];
                  const wt = w[COMPONENT_WEIGHT_KEY[k]];
                  return (
                    <tr key={k}>
                      <td><span className="row-gap"><i style={{ width: 9, height: 9, borderRadius: 2, background: COMPONENT_COLOR[k], display: "inline-block" }} />{k}</span></td>
                      <td className="r">{v === undefined ? <span className="muted" title="Cross-encoder runs on the shortlist only">n/a</span> : v.toFixed(3)}</td>
                      <td className="r">{Math.round(wt * 100)}%</td>
                      <td className="r">{v === undefined ? "—" : (v * wt).toFixed(3)}</td>
                      <td><div className="stack"><span style={{ width: `${(v ?? 0) * 100}%`, background: COMPONENT_COLOR[k] }} /></div></td>
                    </tr>
                  );
                })}
                <tr><td>Honeypot multiplier</td><td className="r">×{res.hp[idx].toFixed(2)}</td><td colSpan={3} className="cell-sub">{res.hp[idx] < 1 ? "Penalty applied" : "No penalty"}</td></tr>
                <tr><td>Disqualifier multiplier</td><td className="r">×{res.dq[idx].toFixed(2)}</td><td colSpan={3} className="cell-sub">{res.dq[idx] < 1 ? "Penalty applied" : "No penalty"}</td></tr>
              </tbody>
            </table>
          </section>

          <section className="drawer-sec">
            <h3>JD-fit breakdown</h3>
            <BarList color={COMPONENT_COLOR["JD-fit"]} max={1}
                     items={Object.keys(cfg.weights.structured).map((k) => ({
                       label: `${titleCase(k)} (${Math.round(cfg.weights.structured[k] * 100)}%)`, value: subs[k] ?? 0,
                     }))} />
          </section>

          {(res.hpFlags[idx].length > 0 || res.dqFlags[idx].length > 0) && (
            <section className="drawer-sec">
              <h3>Trust findings</h3>
              {res.hpFlags[idx].map((f) => (
                <div key={f} className="flag-item crit"><AlertOctagon size={15} style={{ flex: "none", marginTop: 2 }} />
                  <div><b>{flagLabel(f)}</b>{flagDetail(f)}</div></div>
              ))}
              {res.dqFlags[idx].map((f) => (
                <div key={f} className="flag-item warn"><AlertTriangle size={15} style={{ flex: "none", marginTop: 2 }} />
                  <div><b>{flagLabel(f)}</b>{flagDetail(f)}</div></div>
              ))}
            </section>
          )}

          {err && <div className="callout callout-crit"><AlertTriangle size={15} /><span>{err}</span></div>}
          {!detail && !err && <DetailSkeleton />}
          {detail && <Detail d={detail} />}
        </div>
      </aside>
    </>
  );
}

function DetailSkeleton() {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
      {[70, 100, 90, 60].map((w, i) => <div key={i} className="skeleton" style={{ height: 14, width: `${w}%` }} />)}
    </div>
  );
}

function Detail({ d }: { d: CandidateDetail }) {
  const s = d.signals;
  const sig: [string, string][] = [
    ["Recruiter response", fmtPct(s.recruiter_response_rate as number)],
    ["Interview completion", fmtPct(s.interview_completion_rate as number)],
    ["Offer acceptance", fmtPct(s.offer_acceptance_rate as number)],
    ["Avg. response time", s.avg_response_time_hours != null ? `${fmtNum(s.avg_response_time_hours as number, 0)} h` : "—"],
    ["Last active", String(s.last_active_date ?? "—")],
    ["Work mode", titleCase(String(s.preferred_work_mode ?? "—"))],
    ["Willing to relocate", s.willing_to_relocate ? "Yes" : "No"],
    ["GitHub activity", (s.github_activity_score as number) >= 0 ? fmtNum(s.github_activity_score as number, 0) : "—"],
    ["Verified", [s.verified_email && "email", s.verified_phone && "phone"].filter(Boolean).join(", ") || "None"],
  ];
  return (
    <>
      {d.profile.summary && (
        <section className="drawer-sec"><h3>Summary</h3><p style={{ color: "var(--ink-2)", fontSize: 13.5, lineHeight: 1.6 }}>{d.profile.summary}</p></section>
      )}

      <section className="drawer-sec">
        <h3>Career history</h3>
        <div className="career">
          {d.career.map((j, k) => (
            <div key={k} className={`career-item${j.is_current ? " current" : ""}`}>
              <span className="career-dot" />
              <div>
                <div className="career-title">{j.title} · {j.company}</div>
                <div className="career-meta">{j.start_date ?? "?"} – {j.is_current ? "present" : j.end_date ?? "?"} · {j.duration_months ?? "?"} months · {j.industry}{j.company_size && ` · ${j.company_size}`}</div>
                {j.description && <div className="career-desc">{j.description}</div>}
              </div>
            </div>
          ))}
        </div>
      </section>

      <section className="drawer-sec">
        <h3>Skills: claims versus evidence</h3>
        <div className="table-wrap" style={{ border: "1px solid var(--border)", borderRadius: 10 }}>
          <table className="tbl compact">
            <thead><tr><th>Skill</th><th>Proficiency</th><th className="r">Months</th><th className="r">Endorsed</th><th className="r">Assessed</th></tr></thead>
            <tbody>{d.skills.map((s) => (
              <tr key={s.name}>
                <td><span className="row-gap">{s.name}{s.tier1 && <span className="badge badge-brand" style={{ height: 18 }}>JD core</span>}</span></td>
                <td>{titleCase(s.proficiency ?? "—")}</td>
                <td className="r">{s.months ?? "—"}</td>
                <td className="r">{s.endorsements ?? "—"}</td>
                <td className="r">{s.assessed ?? "—"}</td>
              </tr>
            ))}</tbody>
          </table>
        </div>
      </section>

      <section className="drawer-sec">
        <h3>Platform signals</h3>
        <dl className="kv">{sig.map(([k, v]) => <div key={k}><dt>{k}</dt><dd>{v}</dd></div>)}</dl>
      </section>

      {d.education.length > 0 && (
        <section className="drawer-sec">
          <h3>Education</h3>
          {d.education.map((e, k) => (
            <div key={k} style={{ fontSize: 13, marginBottom: 6 }}>
              <b style={{ fontWeight: 600 }}>{e.degree} {e.field_of_study && `· ${e.field_of_study}`}</b>
              <span className="muted"> · {e.institution}{e.end_year && ` · ${e.end_year}`}{e.tier && ` · ${e.tier.replace("_", " ")}`}</span>
            </div>
          ))}
        </section>
      )}

      <section className="drawer-sec">
        <details className="disclose">
          <summary><ChevronRight size={14} />What the models read for this candidate</summary>
          <div className="textbox mt-12">{d.model_text}</div>
          <div className="panel-title mt-12">Job-description terms matched by keyword search ({d.matched_terms.length})</div>
          <div className="tags">{d.matched_terms.map((t) => <span key={t} className="tag">{t}</span>)}</div>
        </details>
      </section>
    </>
  );
}
