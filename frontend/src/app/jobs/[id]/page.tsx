"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { useCallback, useEffect, useState } from "react";
import { api, type JobDetail, type Meta } from "@/lib/api";
import { Overview } from "@/components/Overview";
import { Review, type ReviewFilters } from "@/components/Review";
import { Metrics } from "@/components/Metrics";
import { StatusChip } from "@/components/ui";

type Tab = "overview" | "review" | "metrics";

export default function JobPage() {
  const { id } = useParams<{ id: string }>();
  const [job, setJob] = useState<JobDetail | null>(null);
  const [meta, setMeta] = useState<Meta | null>(null);
  const [tab, setTab] = useState<Tab>("overview");
  const [filters, setFilters] = useState<ReviewFilters>({ task: "agri" });
  const [error, setError] = useState("");

  const load = useCallback(() => api.job(id).then(setJob).catch((e) => setError(e.message)), [id]);

  useEffect(() => {
    load();
    api.meta().then(setMeta).catch((e) => setError(e.message));
  }, [load]);

  const running = job?.status === "queued" || job?.status === "running";
  useEffect(() => {
    if (!running) return;
    const t = setInterval(load, 1500);
    return () => clearInterval(t);
  }, [running, load]);

  // Refresh summary counts when leaving the review tab.
  useEffect(() => {
    if (tab !== "review") load();
  }, [tab, load]);

  if (error) return <p className="text-sm text-bad">{error}</p>;
  if (!job || !meta) return <p className="text-sm text-muted">Loading…</p>;

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-center gap-3">
        <Link href="/" className="text-sm text-muted hover:underline">← Runs</Link>
        <h1 className="text-xl font-semibold">{job.name}</h1>
        <StatusChip status={job.status} />
        {running && <span className="text-sm text-muted">{job.n_docs}/{job.n_files} files processed…</span>}
        <span className="flex-1" />
        <a href={api.exportUrl(job.job_id)} className="rounded border border-border bg-surface px-3 py-1 text-sm hover:bg-surface-2">
          Export JSONL
        </a>
      </div>

      <nav className="flex gap-1 border-b border-border">
        {(["overview", "review", "metrics"] as Tab[]).map((t) => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={`-mb-px border-b-2 px-3 py-2 text-sm capitalize ${
              tab === t ? "border-accent font-medium" : "border-transparent text-muted hover:text-text"
            }`}
          >
            {t}
          </button>
        ))}
      </nav>

      {tab === "overview" && (
        <Overview job={job} onOpenDoc={(doc_id) => { setFilters({ task: filters.task, doc_id }); setTab("review"); }} />
      )}
      {/* Keyed on filters so paging resets when they change. */}
      {tab === "review" && <Review key={JSON.stringify(filters)} job={job} meta={meta} filters={filters} setFilters={setFilters} />}
      {tab === "metrics" && (
        <Metrics
          jobId={job.job_id}
          onInspect={(task, gold, predicted) => {
            setFilters({ task, gold, predicted });
            setTab("review");
          }}
        />
      )}
    </div>
  );
}
