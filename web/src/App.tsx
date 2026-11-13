import { AlertTriangle, Download, GitBranch, ListOrdered, RotateCcw, ShieldAlert, SlidersHorizontal } from "lucide-react";
import { useCallback, useEffect, useMemo, useState } from "react";
import { api, type AppConfig } from "./lib/api";
import { deriveResults } from "./lib/derive";
import { fmtInt, fmtTime } from "./lib/format";
import { STAGE_ORDER, useRun } from "./lib/useRun";
import { CandidateDrawer } from "./components/CandidateDrawer";
import { PipelineView } from "./views/PipelineView";
import { ExportView, RiskView, ShortlistView } from "./views/ResultsViews";
import { SetupView, initialForm, type SetupForm } from "./views/SetupView";

type View = "setup" | "pipeline" | "shortlist" | "risk" | "export";

const VIEW_TITLE: Record<View, string> = {
  setup: "New search", pipeline: "Pipeline", shortlist: "Ranked shortlist", risk: "Risk review", export: "Export",
};

export default function App() {
  const [cfg, setCfg] = useState<AppConfig | null>(null);
  const [cfgErr, setCfgErr] = useState<string | null>(null);
  const [form, setForm] = useState<SetupForm | null>(null);
  const [view, setView] = useState<View>("setup");
  const [drawer, setDrawer] = useState<number | null>(null);
  const { state: run, start, reset } = useRun();

  const loadConfig = useCallback(() => {
    setCfgErr(null);
    api.config().then((c) => { setCfg(c); setForm((f) => f ?? initialForm(c)); })
      .catch((e) => setCfgErr(e instanceof Error ? e.message : String(e)));
  }, []);
  useEffect(loadConfig, [loadConfig]);

  const results = useMemo(() => (run.status === "done" ? deriveResults(run.stages) : null), [run.status, run.stages]);

  const onRun = () => {
    if (!cfg || !form) return;
    const src = cfg.sources.find((s) => s.id === form.source);
    const preset = cfg.jd_presets.find((p) => p.id === form.jdPreset);
    setDrawer(null);
    setView("pipeline");
    void start({
      source: form.source, sample_n: form.sampleN, seed: form.seed, upload_token: form.upload?.token ?? null,
      jd_text: form.jdText, shortlist_k: form.shortlistK, final_n: form.finalN, use_cross_encoder: form.useCE,
    }, {
      sourceLabel: form.source === "upload" ? `Upload · ${form.upload?.filename}` : src?.label ?? form.source,
      jdLabel: preset ? `${preset.label} job description` : "Custom job description",
    }, form.pacing);
  };

  const newSearch = () => { reset(); setDrawer(null); setView("setup"); };

  if (cfgErr) {
    return (
      <div className="empty" style={{ paddingTop: 120 }}>
        <AlertTriangle size={28} color="var(--crit)" />
        <h3 style={{ marginTop: 10 }}>Cannot reach the TalentBridge API</h3>
        <p>{cfgErr}</p>
        <p className="mt-12 mono">uvicorn api.server:app --port 8000</p>
        <button className="btn mt-12" onClick={loadConfig}>Retry</button>
      </div>
    );
  }
  if (!cfg || !form) {
    return <div className="empty" style={{ paddingTop: 160 }}><span className="spinner" style={{ display: "inline-block", width: 22, height: 22 }} /></div>;
  }

  const hasRun = run.status !== "idle";
  const done = results !== null;
  const nav: { id: View; label: string; icon: typeof GitBranch; enabled: boolean; count?: number }[] = [
    { id: "setup", label: "New search", icon: SlidersHorizontal, enabled: true },
    { id: "pipeline", label: "Pipeline", icon: GitBranch, enabled: hasRun },
    { id: "shortlist", label: "Shortlist", icon: ListOrdered, enabled: done, count: results?.rows.length },
    { id: "risk", label: "Risk review", icon: ShieldAlert, enabled: done,
      count: results ? results.hp.filter((v, i) => v < 1 || results.dq[i] < 1).length : undefined },
    { id: "export", label: "Export", icon: Download, enabled: done },
  ];
  const doneStages = STAGE_ORDER.filter((k) => run.stages[k]).length;
  const compute = STAGE_ORDER.reduce((a, k) => a + (run.stages[k]?.seconds ?? 0), 0);

  return (
    <div className="shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">
            <svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true"><path d="M2.5 4h11M8 4v9" stroke="#fff" strokeWidth="2.2" strokeLinecap="round" /><circle cx="12.5" cy="11.5" r="1.8" fill="#9cc1e0" /></svg>
          </div>
          <div><div className="brand-name">TalentBridge</div><div className="brand-sub">Candidate intelligence</div></div>
        </div>
        <div className="nav-label">Workspace</div>
        <nav className="nav">
          {nav.map((n) => (
            <button key={n.id} className={`nav-item${view === n.id ? " active" : ""}`} disabled={!n.enabled} onClick={() => setView(n.id)}>
              <n.icon size={16} strokeWidth={1.9} />{n.label}
              {n.count !== undefined && <span className="count">{fmtInt(n.count)}</span>}
            </button>
          ))}
        </nav>
        <dl className="sidebar-foot">
          <dt>Retrieval</dt><dd>{cfg.models.bi_encoder.split("/").pop()}</dd>
          <dt>Reranker</dt><dd>{cfg.models.cross_encoder.split("/").pop()}</dd>
          <dt>Runtime</dt><dd>CPU · offline models</dd>
        </dl>
      </aside>

      <div className="main">
        <header className="topbar">
          <div className="crumbs">TalentBridge <span>/</span> <b>{VIEW_TITLE[view]}</b></div>
          <div className="topbar-right">
            {hasRun && (
              <span className="badge badge-outline" style={{ height: 28, padding: "0 10px" }}>
                <span className={`status-dot ${run.status}`} />
                {run.status === "running" && `Running · stage ${Math.min(doneStages + 1, 9)} of 9`}
                {run.status === "done" && `Completed · ${fmtInt(run.n ?? 0)} profiles · ${fmtTime(compute)}`}
                {run.status === "error" && "Failed"}
              </span>
            )}
            {hasRun && <button className="btn" onClick={newSearch}><RotateCcw size={14} /> New search</button>}
          </div>
        </header>

        <main className="content">
          {view === "setup" && (
            <SetupView cfg={cfg} form={form} setForm={setForm} onRun={onRun} busy={run.status === "running"} />
          )}
          {view === "pipeline" && hasRun && (
            <PipelineView cfg={cfg} run={run} onOpen={(i) => done && setDrawer(i)} onShortlist={() => setView("shortlist")} />
          )}
          {view === "shortlist" && results && <ShortlistView cfg={cfg} res={results} onOpen={setDrawer} selected={drawer} />}
          {view === "risk" && results && <RiskView res={results} onOpen={setDrawer} />}
          {view === "export" && results && <ExportView cfg={cfg} res={results} run={run} />}
        </main>
      </div>

      {drawer !== null && results && run.runId && (
        <CandidateDrawer cfg={cfg} runId={run.runId} idx={drawer} res={results} onClose={() => setDrawer(null)} />
      )}
    </div>
  );
}
