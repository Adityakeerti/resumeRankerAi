import { ArrowDown, ArrowUp, Minus } from "lucide-react";
import type { ReactNode } from "react";

export function Segmented<T extends string>({
  value, options, onChange, ariaLabel,
}: { value: T; options: { value: T; label: string }[]; onChange: (v: T) => void; ariaLabel: string }) {
  return (
    <div className="seg" role="tablist" aria-label={ariaLabel}>
      {options.map((o) => (
        <button key={o.value} role="tab" aria-selected={value === o.value}
                className={value === o.value ? "on" : ""} onClick={() => onChange(o.value)}>
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function Switch({
  checked, onChange, title, hint,
}: { checked: boolean; onChange: (v: boolean) => void; title: string; hint?: string }) {
  return (
    <label className="switch">
      <input type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} />
      <span className="track" />
      <span className="switch-text"><b>{title}</b>{hint && <span>{hint}</span>}</span>
    </label>
  );
}

export function Slider({
  label, value, min, max, step = 1, onChange, hint, format = String,
}: { label: string; value: number; min: number; max: number; step?: number;
     onChange: (v: number) => void; hint?: ReactNode; format?: (v: number) => string }) {
  return (
    <div className="field">
      <span className="field-label">{label}<span className="val">{format(value)}</span></span>
      <input type="range" min={min} max={max} step={step} value={value} aria-label={label}
             onChange={(e) => onChange(Number(e.target.value))} />
      {hint && <span className="field-hint">{hint}</span>}
    </div>
  );
}

export function Delta({ from, to, enabled = true }: { from: number; to: number; enabled?: boolean }) {
  if (!enabled) return <span className="delta same">—</span>;
  const d = from - to;
  if (d > 0) return <span className="delta up" title={`Up ${d} after reranking`}><ArrowUp size={12} strokeWidth={2.5} />{d}</span>;
  if (d < 0) return <span className="delta down" title={`Down ${-d} after reranking`}><ArrowDown size={12} strokeWidth={2.5} />{-d}</span>;
  return <span className="delta same" title="Unchanged"><Minus size={12} /></span>;
}

export function Card({
  title, sub, step, right, children, foot, bodyClass = "card-body", className = "",
}: { title?: ReactNode; sub?: ReactNode; step?: number; right?: ReactNode; children: ReactNode;
     foot?: ReactNode; bodyClass?: string; className?: string }) {
  return (
    <section className={`card ${className}`}>
      {title && (
        <header className="card-head">
          {step !== undefined && <span className="step-no">{step}</span>}
          <h2>{title}</h2>
          {sub && <span className="sub">{sub}</span>}
          {right && <div className="right">{right}</div>}
        </header>
      )}
      <div className={bodyClass}>{children}</div>
      {foot && <footer className="card-foot">{foot}</footer>}
    </section>
  );
}
