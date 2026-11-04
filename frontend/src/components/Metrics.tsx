"use client";

import { useEffect, useState } from "react";
import { api, pretty, type Report } from "@/lib/api";
import { Card, Stat } from "./ui";

const pct = (v: number | null) => (v == null ? "—" : `${(v * 100).toFixed(1)}%`);

export function Metrics({ jobId, onInspect }: { jobId: string; onInspect: (task: string, gold: string, predicted: string) => void }) {
  const [data, setData] = useState<Record<string, Report> | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    api.metrics(jobId).then(setData).catch((e) => setError(e.message));
  }, [jobId]);

  if (error) return <p className="text-sm text-bad">{error}</p>;
  if (!data) return <p className="text-sm text-muted">Loading…</p>;

  return (
    <div className="space-y-6">
      <p className="text-sm text-muted">
        Predictions are scored against the labels you set in the Review tab. Click a confusion-matrix cell to open those
        sentences.
      </p>
      {Object.entries(data).map(([task, r]) => (
        <Card key={task} title={pretty(task)}>
          {r.n === 0 ? (
            <p className="text-sm text-muted">No human labels for this task yet.</p>
          ) : (
            <div className="space-y-4">
              <div className="grid grid-cols-3 gap-3">
                <Stat label="Labelled" value={r.n} />
                <Stat label="Accuracy" value={pct(r.accuracy)} />
                <Stat label="Macro F1" value={pct(r.macro_f1)} />
              </div>
              <div className="grid lg:grid-cols-2 gap-4">
                <div className="overflow-x-auto">
                  <table className="w-full text-sm tabular-nums">
                    <thead className="text-left text-muted border-b border-border">
                      <tr>
                        <th className="py-1.5 pr-3 font-medium">Class</th>
                        <th className="py-1.5 px-2 font-medium text-right">Precision</th>
                        <th className="py-1.5 px-2 font-medium text-right">Recall</th>
                        <th className="py-1.5 px-2 font-medium text-right">F1</th>
                        <th className="py-1.5 pl-2 font-medium text-right">Support</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-border">
                      {r.labels.map((l) => {
                        const c = r.per_class[l];
                        return (
                          <tr key={l}>
                            <td className="py-1.5 pr-3">{pretty(l)}</td>
                            <td className="py-1.5 px-2 text-right">{pct(c.precision)}</td>
                            <td className="py-1.5 px-2 text-right">{pct(c.recall)}</td>
                            <td className="py-1.5 px-2 text-right font-medium">{pct(c.f1)}</td>
                            <td className="py-1.5 pl-2 text-right text-muted">{c.support}</td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
                <Confusion r={r} onCell={(gold, pred) => onInspect(task, gold, pred)} />
              </div>
            </div>
          )}
        </Card>
      ))}
    </div>
  );
}

function Confusion({ r, onCell }: { r: Report; onCell: (gold: string, pred: string) => void }) {
  const max = Math.max(1, ...r.labels.flatMap((g) => r.labels.map((p) => r.confusion[g][p])));
  return (
    <div className="overflow-x-auto">
      <table className="text-xs tabular-nums">
        <thead>
          <tr>
            <th className="p-1 text-left text-muted font-normal">gold ↓ / pred →</th>
            {r.labels.map((l) => (
              <th key={l} className="p-1 font-medium text-muted max-w-20 truncate" title={l}>{pretty(l)}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {r.labels.map((g) => (
            <tr key={g}>
              <th className="p-1 pr-2 text-left font-medium text-muted whitespace-nowrap">{pretty(g)}</th>
              {r.labels.map((p) => {
                const v = r.confusion[g][p];
                const diag = g === p;
                return (
                  <td key={p} className="p-0.5">
                    <button
                      onClick={() => onCell(g, p)}
                      disabled={!v}
                      title={`gold ${g}, predicted ${p}: ${v}`}
                      className="h-9 min-w-11 w-full rounded"
                      style={{
                        background: v
                          ? `color-mix(in srgb, var(${diag ? "--accent" : "--bad"}) ${15 + (v / max) * 60}%, transparent)`
                          : "var(--surface-2)",
                      }}
                    >
                      {v || ""}
                    </button>
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
