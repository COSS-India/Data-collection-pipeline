"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, LANG_NAMES, pretty, type JobDetail, type Meta, type Sentence } from "@/lib/api";
import { Chip } from "./ui";

export type ReviewFilters = {
  task: string;
  doc_id?: string;
  lang?: string;
  usable?: string;
  predicted?: string;
  gold?: string;
  annotated?: string;
  disagree?: string;
  q?: string;
};

const PAGE_SIZE = 25;

export function Review({
  job,
  meta,
  filters,
  setFilters,
}: {
  job: JobDetail;
  meta: Meta;
  filters: ReviewFilters;
  setFilters: (f: ReviewFilters) => void;
}) {
  const [items, setItems] = useState<Sentence[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [active, setActive] = useState(0);
  const [error, setError] = useState("");
  const [search, setSearch] = useState(filters.q ?? "");
  const listRef = useRef<HTMLOListElement>(null);
  const labels = useMemo(() => meta.tasks[filters.task] ?? [], [meta, filters.task]);

  const load = useCallback(() => {
    api
      .sentences(job.job_id, { ...filters, page, page_size: PAGE_SIZE })
      .then((r) => {
        setItems(r.items);
        setTotal(r.total);
        setError("");
      })
      .catch((e) => setError(e.message));
  }, [job.job_id, filters, page]);

  useEffect(load, [load]);

  const setLabel = useCallback(
    async (s: Sentence, label: string | null) => {
      const task = filters.task;
      // Optimistic update so keyboard labelling stays fast.
      setItems((prev) =>
        prev.map((it) => {
          if (it.sent_id !== s.sent_id) return it;
          const annotations = { ...it.annotations };
          if (label) annotations[task] = { label, note: "", annotator: "" };
          else delete annotations[task];
          return { ...it, annotations };
        }),
      );
      try {
        await api.annotate(s.sent_id, task, label);
      } catch (e) {
        setError((e as Error).message);
        load();
      }
    },
    [filters.task, load],
  );

  // Keyboard: j/k move, 1-9 set label, a accept prediction, x clear, n/p page.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement;
      if (t.tagName === "INPUT" || t.tagName === "SELECT" || t.tagName === "TEXTAREA" || e.metaKey || e.ctrlKey) return;
      const s = items[active];
      if (e.key === "j" || e.key === "ArrowDown") setActive((a) => Math.min(a + 1, items.length - 1));
      else if (e.key === "k" || e.key === "ArrowUp") setActive((a) => Math.max(a - 1, 0));
      else if (e.key === "n" && page * PAGE_SIZE < total) { setPage(page + 1); setActive(0); }
      else if (e.key === "p" && page > 1) { setPage(page - 1); setActive(0); }
      else if (!s) return;
      else if (/^[1-9]$/.test(e.key) && labels[+e.key - 1]) {
        setLabel(s, labels[+e.key - 1]);
        setActive((a) => Math.min(a + 1, items.length - 1));
      } else if (e.key === "a" && s.labels[filters.task]) {
        setLabel(s, s.labels[filters.task].label);
        setActive((a) => Math.min(a + 1, items.length - 1));
      } else if (e.key === "x") setLabel(s, null);
      else return;
      e.preventDefault();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [items, active, labels, filters.task, page, total, setLabel]);

  useEffect(() => {
    listRef.current?.children[active]?.scrollIntoView({ block: "nearest" });
  }, [active]);

  const set = (k: keyof ReviewFilters, v: string) => setFilters({ ...filters, [k]: v || undefined });
  const select = "rounded border border-border bg-surface px-2 py-1 text-sm";

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-2">
        <label className="text-sm font-medium">Task</label>
        <select
          className={select}
          value={filters.task}
          onChange={(e) => setFilters({ ...filters, task: e.target.value, predicted: undefined, gold: undefined })}
        >
          {Object.keys(meta.tasks).map((t) => <option key={t} value={t}>{pretty(t)}</option>)}
        </select>
        <select className={select} value={filters.predicted ?? ""} onChange={(e) => set("predicted", e.target.value)}>
          <option value="">Any prediction</option>
          {labels.map((l) => <option key={l} value={l}>pred: {pretty(l)}</option>)}
        </select>
        <select className={select} value={filters.annotated ?? ""} onChange={(e) => set("annotated", e.target.value)}>
          <option value="">Labelled + unlabelled</option>
          <option value="false">Unlabelled only</option>
          <option value="true">Labelled only</option>
        </select>
        {filters.gold && (
          <button onClick={() => set("gold", "")} className="rounded bg-info-soft px-2 py-1 text-xs text-info">
            gold: {pretty(filters.gold)} ×
          </button>
        )}
        <label className="flex items-center gap-1 text-sm">
          <input type="checkbox" checked={filters.disagree === "true"} onChange={(e) => set("disagree", e.target.checked ? "true" : "")} />
          Errors only
        </label>
        <select className={select} value={filters.lang ?? ""} onChange={(e) => set("lang", e.target.value)}>
          <option value="">All languages</option>
          {Object.keys(job.summary.lang).map((l) => <option key={l} value={l}>{LANG_NAMES[l] ?? l}</option>)}
        </select>
        <select className={select} value={filters.usable ?? ""} onChange={(e) => set("usable", e.target.value)}>
          <option value="">Any quality</option>
          <option value="true">Usable</option>
          <option value="false">Unusable</option>
        </select>
        <select className={`${select} max-w-48`} value={filters.doc_id ?? ""} onChange={(e) => set("doc_id", e.target.value)}>
          <option value="">All documents</option>
          {job.documents.map((d) => <option key={d.doc_id} value={d.doc_id}>{d.source_path}</option>)}
        </select>
        <form onSubmit={(e) => { e.preventDefault(); set("q", search); }} className="flex-1 min-w-40">
          <input
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search text…"
            className="w-full rounded border border-border bg-surface px-2 py-1 text-sm"
          />
        </form>
      </div>

      <div className="flex flex-wrap items-center justify-between gap-2 text-xs text-muted">
        <span>
          {total.toLocaleString()} sentences · page {page} of {Math.max(1, Math.ceil(total / PAGE_SIZE))}
        </span>
        <span className="flex flex-wrap gap-x-3 gap-y-1">
          <span><kbd>j</kbd>/<kbd>k</kbd> move</span>
          <span>
            {labels.slice(0, 9).map((l, i) => <span key={l} className="mr-1.5"><kbd>{i + 1}</kbd> {pretty(l)}</span>)}
          </span>
          <span><kbd>a</kbd> accept prediction</span>
          <span><kbd>x</kbd> clear</span>
          <span><kbd>n</kbd>/<kbd>p</kbd> page</span>
        </span>
      </div>

      {error && <p className="text-sm text-bad">{error}</p>}

      <ol ref={listRef} className="space-y-2">
        {items.map((s, i) => {
          const pred = s.labels[filters.task];
          const gold = s.annotations[filters.task]?.label;
          const verdict = gold ? (gold === pred?.label ? "correct" : "wrong") : null;
          return (
            <li
              key={s.sent_id}
              onClick={() => setActive(i)}
              className={`rounded-lg border bg-surface p-3 transition-colors ${
                i === active ? "border-accent ring-1 ring-accent" : "border-border"
              }`}
            >
              <p lang={s.lang.startsWith("und") ? undefined : s.lang} className="font-deva text-[1.05rem] leading-relaxed">
                {s.text}
              </p>
              <div className="mt-2 flex flex-wrap items-center gap-1.5 text-xs text-muted">
                <span className="truncate max-w-64" title={s.source_path}>{s.source_path}</span>
                {s.location && <span>· {s.location}</span>}
                <span>·</span>
                <Chip tone="info">{LANG_NAMES[s.lang] ?? s.lang}</Chip>
                {!s.quality.usable && <Chip tone="bad">unusable</Chip>}
                {s.quality.duplicate && <Chip tone="warn">duplicate</Chip>}
                {s.quality.flags.filter((f) => f !== "no_terminal_punct").map((f) => <Chip key={f}>{pretty(f)}</Chip>)}
              </div>
              <div className="mt-2 flex flex-wrap items-center gap-2 border-t border-border pt-2">
                <span className="text-xs text-muted">Predicted</span>
                {pred ? (
                  <Chip tone={pred.label === "non_agri" || pred.label === "not_applicable" ? "neutral" : "accent"} title={pred.classifier}>
                    {pretty(pred.label)}
                    {pred.score != null && <span className="ml-1 opacity-70">{pred.score.toFixed(2)}</span>}
                  </Chip>
                ) : (
                  <Chip>none</Chip>
                )}
                {pred?.matched && Object.keys(pred.matched).length > 0 && (
                  <span className="text-xs text-muted">matched: {Object.keys(pred.matched).map(pretty).join(", ")}</span>
                )}
                <span className="flex-1" />
                {verdict && <Chip tone={verdict === "correct" ? "accent" : "bad"}>{verdict}</Chip>}
                <div className="flex flex-wrap gap-1">
                  {labels.map((l, li) => (
                    <button
                      key={l}
                      onClick={(e) => { e.stopPropagation(); setLabel(s, gold === l ? null : l); }}
                      className={`rounded border px-2 py-0.5 text-xs ${
                        gold === l ? "border-accent bg-accent text-white dark:text-black" : "border-border hover:bg-surface-2"
                      }`}
                    >
                      {li < 9 && <span className="opacity-60 mr-1">{li + 1}</span>}
                      {pretty(l)}
                    </button>
                  ))}
                </div>
              </div>
            </li>
          );
        })}
      </ol>

      {items.length === 0 && !error && <p className="text-sm text-muted py-8 text-center">No sentences match these filters.</p>}

      <div className="flex justify-between">
        <button disabled={page === 1} onClick={() => setPage(page - 1)} className="rounded border border-border px-3 py-1 text-sm disabled:opacity-40">
          ← Prev
        </button>
        <button disabled={page * PAGE_SIZE >= total} onClick={() => setPage(page + 1)} className="rounded border border-border px-3 py-1 text-sm disabled:opacity-40">
          Next →
        </button>
      </div>
    </div>
  );
}
