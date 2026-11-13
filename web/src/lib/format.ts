export const fmtTime = (sec: number) =>
  sec < 0.001 ? "<1 ms" : sec < 1 ? `${Math.round(sec * 1000)} ms` : `${sec.toFixed(sec < 10 ? 2 : 1)} s`;

export const fmtInt = (n: number) => n.toLocaleString("en-US");

export const fmtNum = (n: number | null | undefined, d = 2) =>
  n === null || n === undefined || Number.isNaN(n) ? "—" : n.toFixed(d);

export const fmtPct = (n: number | null | undefined, d = 0) =>
  n === null || n === undefined || Number.isNaN(n) || n < 0 ? "—" : `${(n * 100).toFixed(d)}%`;

export const titleCase = (s: string) => s.charAt(0).toUpperCase() + s.slice(1);

export function downloadText(filename: string, text: string, mime = "text/csv") {
  const blob = new Blob([text], { type: `${mime};charset=utf-8` });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export function csvCell(v: unknown): string {
  const s = v === null || v === undefined ? "" : String(v);
  return `"${s.replace(/"/g, '""')}"`;
}
