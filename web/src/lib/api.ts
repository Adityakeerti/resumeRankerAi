// Typed client for api/server.py. Shapes mirror stage_payload() on the server.

export type StageKey =
  | "ingest" | "text" | "semantic" | "lexical" | "structured"
  | "trust" | "stage1" | "rerank" | "final";

export type Component = "Semantic" | "Lexical" | "JD-fit" | "Cross-encoder";

export interface Source { id: string; label: string; description: string; count: number; sampled: boolean }
export interface JdPreset { id: string; label: string; text: string }
export interface StageDef { key: StageKey; title: string; description: string }

export interface AppConfig {
  sources: Source[];
  jd_presets: JdPreset[];
  jd_window: number;
  max_pool: number;
  weights: {
    stage1: Record<string, number>;
    stage2: Record<string, number>;
    structured: Record<string, number>;
  };
  models: { bi_encoder: string; cross_encoder: string };
  stages: StageDef[];
}

export interface CandidateSummary {
  i: number; id: string; name: string; title: string; company: string;
  location: string; yoe: number; notice: number | null; open: boolean; skills: string[];
}

export interface FinalRow {
  rank: number; i: number; id: string; score: number; s1_rank: number;
  penalty: number; contrib: Record<Component, number>; reasoning: string;
}

export interface StageDataMap {
  ingest: { missing: Record<string, number>; dupes: number; yoe: number[]; candidates: CandidateSummary[] };
  text: { avg_chars: number; avg_tokens: number; jd_text: string; jd_terms: number; example_text: string };
  semantic: { dim: number; sem: number[] };
  lexical: { lex: number[]; terms: { term: string; idf: number }[] };
  structured: { fit: number[]; subs: Record<string, number>[] };
  trust: { hp: number[]; dq: number[]; hp_flags: string[][]; dq_flags: string[][] };
  stage1: { k: number; s1: number[]; top: number[] };
  rerank: { enabled: boolean; ce: number[] };
  final: { rows: FinalRow[] };
}

export interface StageEvent<K extends StageKey = StageKey> {
  key: K; seconds: number; headline: string; data: StageDataMap[K];
}

export type StageMap = { [K in StageKey]?: StageEvent<K> };

export type StreamMessage =
  | { type: "start"; run_id: string; n: number }
  | ({ type: "stage" } & StageEvent)
  | { type: "done"; run_id: string; wall_seconds: number; today: string }
  | { type: "error"; message: string };

export interface RunRequest {
  source: string;
  sample_n: number;
  seed: number;
  upload_token: string | null;
  jd_text: string;
  shortlist_k: number;
  final_n: number;
  use_cross_encoder: boolean;
}

export interface UploadResult { token: string; count: number; skipped: number; truncated: boolean; filename: string }

export interface CandidateDetail {
  i: number; id: string;
  profile: Record<string, unknown> & {
    anonymized_name?: string; headline?: string; summary?: string; location?: string; country?: string;
    years_of_experience?: number; current_title?: string; current_company?: string; current_industry?: string;
  };
  career: {
    title: string | null; company: string | null; industry: string | null; company_size: string | null;
    start_date: string | null; end_date: string | null; duration_months: number | null;
    is_current: boolean | null; description: string | null;
  }[];
  education: { institution?: string; degree?: string; field_of_study?: string; end_year?: number; tier?: string }[];
  skills: { name: string; proficiency: string; months: number | null; endorsements: number | null;
            assessed: number | null; tier1: boolean }[];
  signals: Record<string, string | number | boolean | null>;
  model_text: string;
  matched_terms: string[];
  reasoning: string | null;
}

async function asJson<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let msg = `${res.status} ${res.statusText}`;
    try { const body = await res.json(); if (body?.detail) msg = String(body.detail); } catch { /* keep status text */ }
    throw new Error(msg);
  }
  return res.json() as Promise<T>;
}

export const api = {
  config: () => fetch("/api/config").then((r) => asJson<AppConfig>(r)),

  upload: (file: File) => {
    const fd = new FormData();
    fd.append("file", file);
    return fetch("/api/upload", { method: "POST", body: fd }).then((r) => asJson<UploadResult>(r));
  },

  candidate: (runId: string, i: number, signal?: AbortSignal) =>
    fetch(`/api/runs/${runId}/candidates/${i}`, { signal }).then((r) => asJson<CandidateDetail>(r)),

  /** POST /api/run and invoke onMessage for every NDJSON line as it arrives. */
  async run(req: RunRequest, onMessage: (m: StreamMessage) => void, signal: AbortSignal) {
    const res = await fetch("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req),
      signal,
    });
    if (!res.ok || !res.body) await asJson(res);
    const reader = res.body!.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let nl: number;
      while ((nl = buf.indexOf("\n")) >= 0) {
        const line = buf.slice(0, nl).trim();
        buf = buf.slice(nl + 1);
        if (line) onMessage(JSON.parse(line) as StreamMessage);
      }
    }
    if (buf.trim()) onMessage(JSON.parse(buf) as StreamMessage);
  },
};
