import type { CandidateSummary, Component, FinalRow, StageMap } from "./api";

/** Everything a results view needs, joined across stage payloads. Built once per completed run. */
export interface Results {
  n: number;
  cands: CandidateSummary[];
  sem: number[];
  lex: number[];
  fit: number[];
  subs: Record<string, number>[];
  hp: number[];
  dq: number[];
  hpFlags: string[][];
  dqFlags: string[][];
  s1: number[];
  s1Rank: number[];           // 1-based stage-1 rank for every candidate
  shortlist: Set<number>;
  ce: Map<number, number>;    // candidate index → cross-encoder score (shortlist only)
  ceEnabled: boolean;
  rows: FinalRow[];
  finalRank: Map<number, number>;
}

export function deriveResults(st: StageMap): Results | null {
  const { ingest, semantic, lexical, structured, trust, stage1, rerank, final } = st;
  if (!ingest || !semantic || !lexical || !structured || !trust || !stage1 || !rerank || !final) return null;
  const n = ingest.data.candidates.length;
  const s1 = stage1.data.s1;
  const order = s1.map((_, i) => i).sort((a, b) => s1[b] - s1[a]);
  const s1Rank = new Array<number>(n);
  order.forEach((ci, r) => { s1Rank[ci] = r + 1; });
  const ce = new Map<number, number>();
  stage1.data.top.forEach((ci, p) => ce.set(ci, rerank.data.ce[p]));
  return {
    n,
    cands: ingest.data.candidates,
    sem: semantic.data.sem,
    lex: lexical.data.lex,
    fit: structured.data.fit,
    subs: structured.data.subs,
    hp: trust.data.hp,
    dq: trust.data.dq,
    hpFlags: trust.data.hp_flags,
    dqFlags: trust.data.dq_flags,
    s1,
    s1Rank,
    shortlist: new Set(stage1.data.top),
    ce,
    ceEnabled: rerank.data.enabled,
    rows: final.data.rows,
    finalRank: new Map(final.data.rows.map((r) => [r.i, r.rank])),
  };
}

export const COMPONENTS: Component[] = ["Semantic", "Lexical", "JD-fit", "Cross-encoder"];

/** Validated categorical slots (adjacent-pair CVD ΔE ≥ 8, contrast ≥ 3:1 on white). */
export const COMPONENT_COLOR: Record<Component, string> = {
  "Semantic": "#2f6297",
  "Lexical": "#c47c3c",
  "JD-fit": "#13917d",
  "Cross-encoder": "#9a6db8",
};

export const COMPONENT_WEIGHT_KEY: Record<Component, string> = {
  "Semantic": "semantic",
  "Lexical": "lexical",
  "JD-fit": "structured",
  "Cross-encoder": "cross_encoder",
};

export const NEUTRAL = "#b4bcc8";

/** "skill_duration_exceeds_yoe: CNN has 48mo …" → "Skill duration exceeds yoe" */
export function flagLabel(flag: string): string {
  const head = flag.split(":")[0].replace(/_/g, " ").trim();
  return head.charAt(0).toUpperCase() + head.slice(1);
}

export function flagDetail(flag: string): string {
  const i = flag.indexOf(":");
  return i >= 0 ? flag.slice(i + 1).trim() : "";
}

export function topBy(values: number[], k: number): number[] {
  return values.map((_, i) => i).sort((a, b) => values[b] - values[a]).slice(0, k);
}

export function median(xs: number[]): number {
  if (!xs.length) return 0;
  const s = [...xs].sort((a, b) => a - b);
  const m = Math.floor(s.length / 2);
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}
