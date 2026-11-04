import type { ReactNode } from "react";

type Tone = "neutral" | "accent" | "warn" | "bad" | "info";

const TONES: Record<Tone, string> = {
  neutral: "bg-surface-2 text-muted",
  accent: "bg-accent-soft text-accent",
  warn: "bg-warn-soft text-warn",
  bad: "bg-bad-soft text-bad",
  info: "bg-info-soft text-info",
};

export function Chip({ tone = "neutral", children, title }: { tone?: Tone; children: ReactNode; title?: string }) {
  return (
    <span title={title} className={`inline-flex items-center rounded px-1.5 py-0.5 text-xs font-medium ${TONES[tone]}`}>
      {children}
    </span>
  );
}

export function Card({ title, children, className = "" }: { title?: ReactNode; children: ReactNode; className?: string }) {
  return (
    <section className={`rounded-lg border border-border bg-surface ${className}`}>
      {title && <h3 className="border-b border-border px-4 py-2.5 text-sm font-semibold">{title}</h3>}
      <div className="p-4">{children}</div>
    </section>
  );
}

export function Stat({ label, value, sub }: { label: string; value: ReactNode; sub?: ReactNode }) {
  return (
    <div className="rounded-lg border border-border bg-surface px-4 py-3">
      <div className="text-xs uppercase tracking-wide text-muted">{label}</div>
      <div className="mt-1 text-2xl font-semibold tabular-nums">{value}</div>
      {sub && <div className="text-xs text-muted mt-0.5">{sub}</div>}
    </div>
  );
}

export function StatusChip({ status }: { status: string }) {
  const tone: Tone = status === "done" ? "accent" : status === "failed" ? "bad" : "info";
  return <Chip tone={tone}>{status}</Chip>;
}

export function Bars({ data, total }: { data: Record<string, number>; total?: number }) {
  const entries = Object.entries(data).sort((a, b) => b[1] - a[1]);
  const sum = total ?? entries.reduce((s, [, v]) => s + v, 0);
  if (!entries.length) return <p className="text-sm text-muted">No data</p>;
  return (
    <ul className="space-y-1.5">
      {entries.map(([k, v]) => (
        <li key={k} className="grid grid-cols-[9rem_1fr_4rem] items-center gap-2 text-sm">
          <span className="truncate" title={k}>{k.replace(/_/g, " ")}</span>
          <span className="h-2 rounded bg-surface-2 overflow-hidden">
            <span className="block h-full bg-accent" style={{ width: `${sum ? (v / sum) * 100 : 0}%` }} />
          </span>
          <span className="text-right tabular-nums text-muted">{v.toLocaleString()}</span>
        </li>
      ))}
    </ul>
  );
}
