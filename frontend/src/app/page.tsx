"use client";

import Link from "next/link";
import { useCallback, useEffect, useState } from "react";
import { api, type Job, type Meta } from "@/lib/api";
import { Upload } from "@/components/Upload";
import { StatusChip } from "@/components/ui";

export default function Home() {
  const [meta, setMeta] = useState<Meta | null>(null);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [error, setError] = useState("");

  const load = useCallback(() => api.jobs().then(setJobs).catch((e) => setError(e.message)), []);

  useEffect(() => {
    api.meta().then(setMeta).catch((e) => setError(`Backend unreachable: ${e.message}`));
    load();
  }, [load]);

  // Poll while anything is still processing.
  useEffect(() => {
    if (!jobs.some((j) => j.status === "queued" || j.status === "running")) return;
    const t = setInterval(load, 2000);
    return () => clearInterval(t);
  }, [jobs, load]);

  return (
    <div className="space-y-8">
      <section>
        <h1 className="text-xl font-semibold mb-1">New run</h1>
        <p className="text-sm text-muted mb-4">
          Upload source files. Each run extracts text, splits it into sentences, detects language and quality, and
          applies the enabled classifiers.
        </p>
        {meta ? <Upload extensions={meta.extensions} /> : !error && <p className="text-sm text-muted">Loading…</p>}
      </section>

      {error && <p className="rounded border border-bad bg-bad-soft px-3 py-2 text-sm text-bad">{error}</p>}

      <section>
        <h2 className="text-lg font-semibold mb-3">Runs</h2>
        {jobs.length === 0 ? (
          <p className="text-sm text-muted">No runs yet.</p>
        ) : (
          <div className="overflow-x-auto rounded-lg border border-border bg-surface">
            <table className="w-full text-sm">
              <thead className="text-left text-muted border-b border-border">
                <tr>
                  <th className="px-4 py-2 font-medium">Name</th>
                  <th className="px-4 py-2 font-medium">Status</th>
                  <th className="px-4 py-2 font-medium text-right">Files</th>
                  <th className="px-4 py-2 font-medium text-right">Sentences</th>
                  <th className="px-4 py-2 font-medium">Created</th>
                  <th />
                </tr>
              </thead>
              <tbody className="divide-y divide-border">
                {jobs.map((j) => (
                  <tr key={j.job_id} className="hover:bg-surface-2">
                    <td className="px-4 py-2">
                      <Link href={`/jobs/${j.job_id}`} className="font-medium hover:underline">{j.name}</Link>
                      {j.errors.length > 0 && <span className="ml-2 text-xs text-bad">{j.errors.length} error(s)</span>}
                    </td>
                    <td className="px-4 py-2"><StatusChip status={j.status} /></td>
                    <td className="px-4 py-2 text-right tabular-nums">{j.n_docs}/{j.n_files}</td>
                    <td className="px-4 py-2 text-right tabular-nums">{j.n_sentences.toLocaleString()}</td>
                    <td className="px-4 py-2 text-muted">{new Date(j.created_at * 1000).toLocaleString()}</td>
                    <td className="px-4 py-2 text-right">
                      <button
                        className="text-xs text-muted hover:text-bad"
                        onClick={async () => {
                          if (confirm(`Delete run "${j.name}" and its annotations?`)) {
                            await api.deleteJob(j.job_id);
                            load();
                          }
                        }}
                      >
                        Delete
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  );
}
