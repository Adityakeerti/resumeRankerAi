import { useMemo, useState } from "react";
import {
  Bar, BarChart, CartesianGrid, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis, ZAxis,
} from "recharts";
import type { Component } from "../lib/api";
import { COMPONENTS, COMPONENT_COLOR } from "../lib/derive";

const AXIS = { stroke: "#cbd2dc", tick: { fill: "#737f8f", fontSize: 11 }, tickLine: false } as const;
const GRID = { stroke: "#edf0f4", vertical: false } as const;

function Tip({ title, rows }: { title?: string; rows: [string, string][] }) {
  return (
    <div className="chart-tip">
      {title && <div className="t">{title}</div>}
      {rows.map(([k, v]) => <div className="row" key={k}><span>{k}</span><b>{v}</b></div>)}
    </div>
  );
}

/* ── Histogram ─────────────────────────────────────────────────────────── */
export function Histogram({
  values, bins = 24, color = "#2f6297", xLabel, height = 190, digits = 2,
}: { values: number[]; bins?: number; color?: string; xLabel: string; height?: number; digits?: number }) {
  const data = useMemo(() => {
    if (!values.length) return [];
    const lo = Math.min(...values), hi = Math.max(...values);
    const w = (hi - lo) / bins || 1;
    const counts = new Array(bins).fill(0);
    values.forEach((v) => { counts[Math.min(bins - 1, Math.floor((v - lo) / w))]++; });
    return counts.map((c, i) => ({ x0: lo + i * w, x1: lo + (i + 1) * w, mid: lo + (i + 0.5) * w, count: c }));
  }, [values, bins]);
  return (
    <div>
      <ResponsiveContainer width="100%" height={height}>
        <BarChart data={data} margin={{ top: 6, right: 6, bottom: 4, left: -16 }} barCategoryGap={1}>
          <CartesianGrid {...GRID} />
          <XAxis dataKey="mid" {...AXIS} tickFormatter={(v: number) => v.toFixed(digits === 0 ? 0 : 2)} minTickGap={24} />
          <YAxis {...AXIS} axisLine={false} allowDecimals={false} width={44} />
          <Tooltip cursor={{ fill: "rgba(47,84,130,0.06)" }} content={({ active, payload }) => {
            if (!active || !payload?.length) return null;
            const d = payload[0].payload as (typeof data)[number];
            return <Tip title={`${d.x0.toFixed(digits)} – ${d.x1.toFixed(digits)}`} rows={[["Profiles", String(d.count)]]} />;
          }} />
          <Bar dataKey="count" fill={color} radius={[3, 3, 0, 0]} isAnimationActive={false} />
        </BarChart>
      </ResponsiveContainer>
      <div className="muted" style={{ fontSize: 11.5, textAlign: "center", marginTop: -2 }}>{xLabel}</div>
    </div>
  );
}

/* ── Grouped scatter ───────────────────────────────────────────────────── */
export interface ScatterPoint { x: number; y: number; group: string; label: string; sub?: string }
export function GroupScatter({
  points, groups, xLabel, yLabel, height = 340, xFmt = (v: number) => v.toFixed(2), yFmt = (v: number) => v.toFixed(2),
  xDomain, yDomain, reverseX = false,
}: {
  points: ScatterPoint[]; groups: { key: string; label: string; color: string }[];
  xLabel: string; yLabel: string; height?: number;
  xFmt?: (v: number) => string; yFmt?: (v: number) => string;
  xDomain?: [number | "auto", number | "auto"]; yDomain?: [number | "auto", number | "auto"]; reverseX?: boolean;
}) {
  return (
    <div>
      <div className="legend" style={{ marginBottom: 8 }}>
        {groups.map((g) => (
          <span key={g.key}><i style={{ background: g.color, borderRadius: "50%" }} />{g.label}
            <span className="muted"> · {points.filter((p) => p.group === g.key).length}</span></span>
        ))}
      </div>
      <ResponsiveContainer width="100%" height={height}>
        <ScatterChart margin={{ top: 8, right: 12, bottom: 18, left: 0 }}>
          <CartesianGrid stroke="#edf0f4" />
          <XAxis type="number" dataKey="x" name={xLabel} {...AXIS} domain={xDomain ?? ["auto", "auto"]} reversed={reverseX}
                 tickFormatter={xFmt} label={{ value: xLabel, position: "insideBottom", offset: -10, fill: "#737f8f", fontSize: 11.5 }} />
          <YAxis type="number" dataKey="y" name={yLabel} {...AXIS} domain={yDomain ?? ["auto", "auto"]} width={52}
                 tickFormatter={yFmt} label={{ value: yLabel, angle: -90, position: "insideLeft", offset: 12, fill: "#737f8f", fontSize: 11.5, style: { textAnchor: "middle" } }} />
          <ZAxis range={[46, 46]} />
          <Tooltip cursor={{ strokeDasharray: "3 3", stroke: "#cbd2dc" }} content={({ active, payload }) => {
            if (!active || !payload?.length) return null;
            const p = payload[0].payload as ScatterPoint;
            const g = groups.find((x) => x.key === p.group);
            return <Tip title={p.label} rows={[
              ...(p.sub ? [["Role", p.sub] as [string, string]] : []),
              [xLabel, xFmt(p.x)], [yLabel, yFmt(p.y)], ["Group", g?.label ?? p.group],
            ]} />;
          }} />
          {[...groups].reverse().map((g) => (
            <Scatter key={g.key} data={points.filter((p) => p.group === g.key)} fill={g.color}
                     stroke="#fff" strokeWidth={1} fillOpacity={0.85} isAnimationActive={false} />
          ))}
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
}

/* ── Horizontal bar list ───────────────────────────────────────────────── */
export function BarList({
  items, max, fmt = (v: number) => v.toFixed(2), color = "#2f6297",
}: { items: { label: string; value: number; color?: string; title?: string }[]; max?: number;
     fmt?: (v: number) => string; color?: string }) {
  const m = max ?? Math.max(...items.map((i) => i.value), 1e-9);
  return (
    <div className="barlist">
      {items.map((it) => (
        <div className="barlist-row" key={it.label} title={it.title ?? `${it.label}: ${fmt(it.value)}`}>
          <span className="lbl">{it.label}</span>
          <span className="track"><span className="fill" style={{ display: "block", width: `${(it.value / m) * 100}%`, background: it.color ?? color }} /></span>
          <span className="v">{fmt(it.value)}</span>
        </div>
      ))}
    </div>
  );
}

/* ── Heatmap (sequential single hue) ───────────────────────────────────── */
const RAMP: [number, number, number][] = [[233, 239, 247], [29, 63, 102]]; // light → dark navy
function rampColor(t: number) {
  const c = RAMP[0].map((a, i) => Math.round(a + (RAMP[1][i] - a) * Math.max(0, Math.min(1, t))));
  return `rgb(${c.join(",")})`;
}
export function Heatmap({
  rows, cols, rowWidth = 190,
}: { rows: { label: string; values: number[] }[]; cols: { label: string; sub?: string }[]; rowWidth?: number }) {
  return (
    <div className="heat" style={{ gridTemplateColumns: `${rowWidth}px repeat(${cols.length}, minmax(44px, 1fr))` }}>
      <span />
      {cols.map((c) => <span className="heat-col" key={c.label}>{c.label}{c.sub && <><br /><span style={{ color: "#9aa4b2" }}>{c.sub}</span></>}</span>)}
      {rows.map((r) => (
        <Row key={r.label} r={r} cols={cols} />
      ))}
      <div style={{ gridColumn: `1 / span ${cols.length + 1}`, display: "flex", alignItems: "center", gap: 8, marginTop: 8, color: "#737f8f" }}>
        <span>0</span>
        <span style={{ width: 140, height: 8, borderRadius: 4, background: `linear-gradient(90deg, ${rampColor(0)}, ${rampColor(1)})` }} />
        <span>1.0 sub-score</span>
      </div>
    </div>
  );
}
function Row({ r, cols }: { r: { label: string; values: number[] }; cols: { label: string }[] }) {
  return (
    <>
      <span className="heat-row" title={r.label}>{r.label}</span>
      {r.values.map((v, j) => (
        <span key={j} className="heat-cell" title={`${r.label} · ${cols[j].label}: ${v.toFixed(2)}`}
              style={{ background: rampColor(v), color: v > 0.55 ? "#fff" : "#2a3647" }}>{v.toFixed(2)}</span>
      ))}
    </>
  );
}

/* ── Stacked contribution bar ──────────────────────────────────────────── */
export function ContributionBar({ contrib, large = false }: { contrib: Record<Component, number>; large?: boolean }) {
  return (
    <div className={`stack${large ? " lg" : ""}`}
         title={COMPONENTS.map((c) => `${c}: ${contrib[c].toFixed(3)}`).join("\n")}>
      {COMPONENTS.map((c) => contrib[c] > 0 && (
        <span key={c} style={{ width: `${contrib[c] * 100}%`, background: COMPONENT_COLOR[c] }} />
      ))}
    </div>
  );
}

export function ComponentLegend({ weights }: { weights: Record<string, number> }) {
  const key: Record<Component, string> = { "Semantic": "semantic", "Lexical": "lexical", "JD-fit": "structured", "Cross-encoder": "cross_encoder" };
  return (
    <div className="legend">
      {COMPONENTS.map((c) => (
        <span key={c}><i style={{ background: COMPONENT_COLOR[c] }} />{c} <span className="muted">{Math.round((weights[key[c]] ?? 0) * 100)}%</span></span>
      ))}
    </div>
  );
}

/* ── Slope chart: stage-1 rank → final rank ────────────────────────────── */
export function SlopeChart({
  items, height = 420,
}: { items: { id: string; label: string; from: number; to: number }[]; height?: number }) {
  const [hover, setHover] = useState<string | null>(null);
  const maxRank = Math.max(...items.map((d) => Math.max(d.from, d.to)), 2);
  const W = 520, padT = 30, padB = 14, x0 = 70, x1 = W - 70;
  const y = (r: number) => padT + ((r - 1) / (maxRank - 1)) * (height - padT - padB);
  const color = (d: { from: number; to: number }) => (d.to < d.from ? "#2f6297" : d.to > d.from ? "#c47c3c" : "#b4bcc8");
  const hovered = items.find((d) => d.id === hover);
  return (
    <div style={{ position: "relative" }}>
      <div className="legend" style={{ marginBottom: 8 }}>
        <span><i style={{ background: "#2f6297" }} />Moved up</span>
        <span><i style={{ background: "#c47c3c" }} />Moved down</span>
        <span><i style={{ background: "#b4bcc8" }} />Unchanged</span>
      </div>
      <svg viewBox={`0 0 ${W} ${height}`} width="100%" role="img" aria-label="Rank change from stage 1 to final">
        <text x={x0} y={14} textAnchor="middle" fontSize="11.5" fill="#737f8f">Stage-1 rank</text>
        <text x={x1} y={14} textAnchor="middle" fontSize="11.5" fill="#737f8f">Final rank</text>
        <line x1={x0} x2={x0} y1={padT - 6} y2={height - padB + 4} stroke="#e2e6ec" />
        <line x1={x1} x2={x1} y1={padT - 6} y2={height - padB + 4} stroke="#e2e6ec" />
        {items.map((d) => {
          const on = hover === d.id, dim = hover !== null && !on;
          return (
            <g key={d.id} onMouseEnter={() => setHover(d.id)} onMouseLeave={() => setHover(null)} style={{ cursor: "default" }}>
              <line x1={x0} y1={y(d.from)} x2={x1} y2={y(d.to)} stroke="transparent" strokeWidth={10} />
              <line x1={x0} y1={y(d.from)} x2={x1} y2={y(d.to)} stroke={color(d)} strokeWidth={on ? 2.5 : 1.6}
                    opacity={dim ? 0.18 : 0.9} />
              <circle cx={x0} cy={y(d.from)} r={on ? 4.5 : 3.2} fill={color(d)} stroke="#fff" strokeWidth={1.5} opacity={dim ? 0.25 : 1} />
              <circle cx={x1} cy={y(d.to)} r={on ? 4.5 : 3.2} fill={color(d)} stroke="#fff" strokeWidth={1.5} opacity={dim ? 0.25 : 1} />
              {on && (
                <text x={x1 + 9} y={y(d.to) + 4} fontSize="11" fill="#455163">#{d.to} {d.label}</text>
              )}
              {on && <text x={x0 - 9} y={y(d.from) + 4} fontSize="11" fill="#455163" textAnchor="end">#{d.from}</text>}
            </g>
          );
        })}
      </svg>
      {hovered && (
        <div className="chart-tip" style={{ position: "absolute", top: 28, right: 0 }}>
          <div className="t">{hovered.label}</div>
          <div className="row"><span>Stage-1 rank</span><b>{hovered.from}</b></div>
          <div className="row"><span>Final rank</span><b>{hovered.to}</b></div>
          <div className="row"><span>Change</span><b>{hovered.from - hovered.to > 0 ? "+" : ""}{hovered.from - hovered.to}</b></div>
        </div>
      )}
    </div>
  );
}
