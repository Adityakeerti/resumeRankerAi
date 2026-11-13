import { useCallback, useRef, useState } from "react";
import { api, type RunRequest, type StageKey, type StageMap, type StreamMessage } from "./api";

export const STAGE_ORDER: StageKey[] = [
  "ingest", "text", "semantic", "lexical", "structured", "trust", "stage1", "rerank", "final",
];

export interface RunMeta { sourceLabel: string; jdLabel: string; startedAt: number }

export interface RunState {
  status: "idle" | "running" | "done" | "error";
  runId?: string;
  n?: number;
  stages: StageMap;
  active?: StageKey;          // stage currently shown as "running"
  wall?: number;
  today?: string;
  error?: string;
  request?: RunRequest;
  meta?: RunMeta;
}

const IDLE: RunState = { status: "idle", stages: {} };
const PACE_MS = 550;

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

/**
 * Drives one pipeline run. Stream messages are queued and revealed in order;
 * with pacing on, each reveal waits PACE_MS so an audience can follow along.
 * Reported stage timings are always the server's real compute time.
 */
export function useRun() {
  const [state, setState] = useState<RunState>(IDLE);
  const abortRef = useRef<AbortController | null>(null);

  const start = useCallback(async (req: RunRequest, meta: Omit<RunMeta, "startedAt">, pacing: boolean) => {
    abortRef.current?.abort();
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    setState({ status: "running", stages: {}, active: "ingest", request: req, meta: { ...meta, startedAt: Date.now() } });

    let chain = Promise.resolve();
    const enqueue = (fn: () => void, delay: number) => {
      chain = chain.then(async () => {
        if (delay) await sleep(delay);
        if (!ctrl.signal.aborted) fn();
      });
    };

    const onMessage = (m: StreamMessage) => {
      switch (m.type) {
        case "start":
          enqueue(() => setState((s) => ({ ...s, runId: m.run_id, n: m.n })), 0);
          break;
        case "stage": {
          const { type: _t, ...ev } = m;
          void _t;
          enqueue(() => setState((s) => {
            const next = STAGE_ORDER[STAGE_ORDER.indexOf(ev.key) + 1];
            return { ...s, stages: { ...s.stages, [ev.key]: ev }, active: next };
          }), pacing ? PACE_MS : 0);
          break;
        }
        case "done":
          enqueue(() => setState((s) => ({ ...s, status: "done", active: undefined, runId: m.run_id,
                                           wall: m.wall_seconds, today: m.today })), pacing ? 250 : 0);
          break;
        case "error":
          enqueue(() => setState((s) => ({ ...s, status: "error", active: undefined, error: m.message })), 0);
          break;
      }
    };

    try {
      await api.run(req, onMessage, ctrl.signal);
    } catch (e) {
      if (ctrl.signal.aborted) return;
      enqueue(() => setState((s) => ({ ...s, status: "error", active: undefined,
                                        error: e instanceof Error ? e.message : String(e) })), 0);
    }
  }, []);

  const reset = useCallback(() => {
    abortRef.current?.abort();
    setState(IDLE);
  }, []);

  return { state, start, reset };
}
