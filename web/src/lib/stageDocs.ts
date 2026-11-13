import type { StageKey } from "./api";

export const STAGE_DOCS: Record<StageKey, { what: string; why: string }> = {
  ingest: {
    what: "Every profile is parsed and checked for the fields the scorers depend on: career history, skills and platform signals.",
    why: "Missing sections silently lower scores. Surfacing them up-front separates data-quality problems from genuine weak fits.",
  },
  text: {
    what: "Each profile is flattened into one text representation (career descriptions first, then headline, summary, strong skills and field of study), plus a token list for keyword search.",
    why: "Career descriptions carry the most evidence; skill lists are easy to pad. Leading with descriptions puts evidence inside the models' reading window.",
  },
  semantic: {
    what: "A bi-encoder embeds the job description and every profile independently. Cosine similarity measures topical fit.",
    why: "Captures paraphrase — 'vector search' and 'ANN retrieval' land close together — and is cheap enough to run over 100,000 profiles.",
  },
  lexical: {
    what: "BM25 scores exact term overlap between the job description and each profile, weighting terms by how rare they are in the pool.",
    why: "Rare, specific terms such as FAISS, NDCG or Qdrant are strong evidence that embeddings tend to blur.",
  },
  structured: {
    what: "Eight signals derived from the job description's explicit requirements are computed and combined with fixed weights.",
    why: "Most of a job description is constraints — years of experience, product versus services background, location, notice period — that text similarity cannot see.",
  },
  trust: {
    what: "Honeypot checks catch internally inconsistent profiles. Disqualifier rules encode the job description's explicit exclusions.",
    why: "Both apply multiplicative penalties rather than hard filters, so one noisy signal cannot silently remove an otherwise strong candidate.",
  },
  stage1: {
    what: "Semantic, lexical and JD-fit scores are fused, penalties applied, and the top of the pool is kept as a shortlist.",
    why: "The cross-encoder costs roughly 100× more per profile than the bi-encoder, so it only sees candidates who are already plausible.",
  },
  rerank: {
    what: "A cross-encoder reads the job description and each shortlisted profile together and outputs a relevance probability.",
    why: "Joint attention over both texts is more precise than comparing two independently computed embeddings.",
  },
  final: {
    what: "Stage-2 fusion adds the cross-encoder score, re-applies penalties, breaks ties by candidate ID and writes a reason for every rank.",
    why: "Reasons are assembled from facts in the candidate's own record, so they cannot invent qualifications.",
  },
};
