import { AlertTriangle, FileUp, Play, X } from "lucide-react";
import { useRef, useState, type ReactNode } from "react";
import { api, type AppConfig, type UploadResult } from "../lib/api";
import { fmtInt } from "../lib/format";
import { Card, Segmented, Slider, Switch } from "../components/ui";

export interface SetupForm {
  jdPreset: string;               // preset id or "custom"
  jdText: string;
  source: string;                 // source id or "upload"
  sampleN: number;
  seed: number;
  upload: UploadResult | null;
  shortlistK: number;
  finalN: number;
  useCE: boolean;
  pacing: boolean;
}

export function initialForm(cfg: AppConfig): SetupForm {
  const jd = cfg.jd_presets[0];
  return {
    jdPreset: jd?.id ?? "custom", jdText: jd?.text ?? "",
    source: cfg.sources[0]?.id ?? "upload", sampleN: 500, seed: 42, upload: null,
    shortlistK: 50, finalN: 25, useCE: true, pacing: true,
  };
}

export function poolSize(cfg: AppConfig, f: SetupForm): number {
  if (f.source === "upload") return f.upload?.count ?? 0;
  const s = cfg.sources.find((x) => x.id === f.source);
  if (!s) return 0;
  return s.sampled ? f.sampleN : s.count;
}

export function SetupView({
  cfg, form, setForm, onRun, busy,
}: { cfg: AppConfig; form: SetupForm; setForm: (f: SetupForm) => void; onRun: () => void; busy: boolean }) {
  const set = <K extends keyof SetupForm>(k: K, v: SetupForm[K]) => setForm({ ...form, [k]: v });
  const n = poolSize(cfg, form);
  const over = form.jdText.length > cfg.jd_window;
  const k = Math.min(form.shortlistK, n || form.shortlistK);
  const canRun = n > 0 && form.jdText.trim().length >= 20 && !busy;

  const presetOptions = [
    ...cfg.jd_presets.map((p) => ({ value: p.id, label: p.label })),
    { value: "custom", label: "Custom" },
  ];

  return (
    <>
      <div className="page-head">
        <div>
          <h1>New search</h1>
          <p>Describe the role, choose who to screen, and run the ranking pipeline. Every stage is shown as it executes.</p>
        </div>
      </div>

      <div className="setup-grid">
        <Card step={1} title="Role" sub="The job description candidates are ranked against"
              right={<Segmented ariaLabel="Job description source" value={form.jdPreset} options={presetOptions}
                                onChange={(id) => {
                                  const p = cfg.jd_presets.find((x) => x.id === id);
                                  setForm({ ...form, jdPreset: id, jdText: p ? p.text : form.jdPreset === "custom" ? form.jdText : "" });
                                }} />}>
          <textarea className="textarea" value={form.jdText} aria-label="Job description"
                    placeholder="Paste or write the job description…"
                    onChange={(e) => setForm({ ...form, jdText: e.target.value, jdPreset: "custom" })} />
          <div className="field mt-12">
            <div className="row-gap" style={{ fontSize: 12, color: "var(--ink-3)" }}>
              <span className="num">{fmtInt(form.jdText.length)} characters</span>
              <span style={{ marginLeft: "auto" }}>Model reading window: {fmtInt(cfg.jd_window)}</span>
            </div>
            <div className={`window-meter${over ? " over" : ""}`}>
              <span style={{ width: `${Math.min(100, (cfg.jd_window / Math.max(form.jdText.length, 1)) * 100)}%` }} />
            </div>
          </div>
          {over && (
            <div className="callout callout-warn mt-12">
              <AlertTriangle size={15} />
              <span>The embedding models read only the first {fmtInt(cfg.jd_window)} characters
                (about {Math.round((cfg.jd_window / form.jdText.length) * 100)}% of this text). Requirements later in
                the document will not influence semantic scoring. The condensed JD keeps them inside the window.</span>
            </div>
          )}
        </Card>

        <Card step={2} title="Talent pool" sub="Profiles to screen">
          <div className="source-list">
            {cfg.sources.map((s) => (
              <SourceOption key={s.id} on={form.source === s.id} onSelect={() => set("source", s.id)}
                            title={s.label} meta={s.sampled ? `${fmtInt(s.count)} available` : `${fmtInt(s.count)} profiles`}
                            desc={s.description}>
                {s.sampled && form.source === s.id && (
                  <div className="source-extra" onClick={(e) => e.stopPropagation()}>
                    <Slider label="Sample size" value={form.sampleN} min={100} max={cfg.max_pool} step={100}
                            onChange={(v) => set("sampleN", v)} format={fmtInt} />
                    <div className="field">
                      <span className="field-label">Seed</span>
                      <input className="input num" type="number" min={0} max={9999} value={form.seed}
                             onChange={(e) => set("seed", Number(e.target.value) || 0)} aria-label="Random seed" />
                    </div>
                  </div>
                )}
              </SourceOption>
            ))}
            <SourceOption on={form.source === "upload"} onSelect={() => set("source", "upload")}
                          title="Upload a file" meta={form.upload ? `${fmtInt(form.upload.count)} profiles` : "JSONL or JSON"}
                          desc={`Candidate records matching the challenge schema. Up to ${fmtInt(cfg.max_pool)} profiles per run.`}>
              {form.source === "upload" && (
                <Uploader value={form.upload} onChange={(u) => set("upload", u)} />
              )}
            </SourceOption>
          </div>
        </Card>
      </div>

      <Card step={3} title="Pipeline settings" className="mt-12" bodyClass="card-body">
        <div className="settings-grid" style={{ marginTop: 4 }}>
          <Slider label="Stage-1 shortlist" value={form.shortlistK} min={10} max={500} step={10}
                  onChange={(v) => set("shortlistK", v)}
                  hint="Profiles passed to the cross-encoder. Larger is slower but more thorough." />
          <Slider label="Final ranked list" value={form.finalN} min={5} max={100} step={5}
                  onChange={(v) => set("finalN", v)} hint="Profiles ranked, explained and exported." />
          <Switch checked={form.useCE} onChange={(v) => set("useCE", v)} title="Cross-encoder rerank"
                  hint="Joint JD–profile relevance on the shortlist." />
          <Switch checked={form.pacing} onChange={(v) => set("pacing", v)} title="Presentation pacing"
                  hint="Paces stage reveals for an audience. Timings stay real." />
        </div>
      </Card>

      <div className="card runbar">
        <div className="runbar-summary">
          <span>Pool <b>{fmtInt(n)}</b></span>
          <span>Shortlist <b>{fmtInt(k)}</b></span>
          <span>Ranked <b>{fmtInt(Math.min(form.finalN, k))}</b></span>
          <span>Reranking <b>{form.useCE ? "On" : "Off"}</b></span>
        </div>
        <button className="btn btn-primary btn-lg" disabled={!canRun} onClick={onRun}>
          <Play size={15} fill="currentColor" /> Run ranking
        </button>
      </div>
    </>
  );
}

function SourceOption({
  on, onSelect, title, meta, desc, children,
}: { on: boolean; onSelect: () => void; title: string; meta: string; desc: string; children?: ReactNode }) {
  return (
    <div role="radio" aria-checked={on} tabIndex={0} className={`source${on ? " on" : ""}`} onClick={onSelect}
         onKeyDown={(e) => { if (e.key === " " || e.key === "Enter") { e.preventDefault(); onSelect(); } }}>
      <span className="radio" />
      <div style={{ flex: 1, minWidth: 0 }}>
        <div className="source-title">{title}<span className="badge" style={{ marginLeft: "auto" }}>{meta}</span></div>
        <div className="source-desc">{desc}</div>
        {children}
      </div>
    </div>
  );
}

function Uploader({ value, onChange }: { value: UploadResult | null; onChange: (u: UploadResult | null) => void }) {
  const ref = useRef<HTMLInputElement>(null);
  const [drag, setDrag] = useState(false);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const send = async (file: File) => {
    setBusy(true); setErr(null);
    try { onChange(await api.upload(file)); } catch (e) { setErr(e instanceof Error ? e.message : String(e)); onChange(null); }
    finally { setBusy(false); }
  };

  if (value) {
    return (
      <div className="callout callout-info mt-12" onClick={(e) => e.stopPropagation()}>
        <FileUp size={15} />
        <span style={{ flex: 1 }}><b>{value.filename}</b> · {fmtInt(value.count)} profiles loaded
          {value.skipped > 0 && ` · ${value.skipped} malformed rows skipped`}
          {value.truncated && " · truncated to the pool limit"}</span>
        <button className="btn btn-ghost btn-icon" style={{ height: 22 }} aria-label="Remove file" onClick={() => onChange(null)}><X size={14} /></button>
      </div>
    );
  }
  return (
    <div onClick={(e) => e.stopPropagation()}>
      <div className={`drop${drag ? " drag" : ""}`} onClick={() => ref.current?.click()}
           onDragOver={(e) => { e.preventDefault(); setDrag(true); }} onDragLeave={() => setDrag(false)}
           onDrop={(e) => { e.preventDefault(); setDrag(false); const f = e.dataTransfer.files[0]; if (f) void send(f); }}>
        {busy ? <span className="row-gap" style={{ justifyContent: "center" }}><span className="spinner" /> Parsing file…</span>
              : <><b>Choose a file</b> or drag it here</>}
        <input ref={ref} type="file" accept=".jsonl,.json" hidden onChange={(e) => { const f = e.target.files?.[0]; if (f) void send(f); }} />
      </div>
      {err && <div className="callout callout-crit mt-12"><AlertTriangle size={15} /><span>{err}</span></div>}
    </div>
  );
}
