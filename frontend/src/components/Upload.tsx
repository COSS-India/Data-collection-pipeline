"use client";

import { useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { api } from "@/lib/api";

export function Upload({ extensions }: { extensions: string[] }) {
  const router = useRouter();
  const fileInput = useRef<HTMLInputElement>(null);
  const dirInput = useRef<HTMLInputElement>(null);
  const [files, setFiles] = useState<File[]>([]);
  const [name, setName] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [drag, setDrag] = useState(false);

  const accept = (list: FileList | null) => {
    if (!list) return;
    const ok = Array.from(list).filter((f) => extensions.some((e) => f.name.toLowerCase().endsWith(e)));
    setFiles((prev) => [...prev, ...ok]);
    if (ok.length < list.length) setError(`Skipped ${list.length - ok.length} unsupported file(s).`);
  };

  const submit = async () => {
    setBusy(true);
    setError("");
    try {
      const res = await api.createJob(files, name);
      router.push(`/jobs/${res.job_id}`);
    } catch (e) {
      setError((e as Error).message);
      setBusy(false);
    }
  };

  return (
    <div className="space-y-3">
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDrag(true);
        }}
        onDragLeave={() => setDrag(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDrag(false);
          accept(e.dataTransfer.files);
        }}
        className={`rounded-lg border-2 border-dashed px-6 py-8 text-center transition-colors ${
          drag ? "border-accent bg-accent-soft" : "border-border bg-surface"
        }`}
      >
        <p className="font-medium">Drop files here</p>
        <p className="mt-1 text-sm text-muted">
          or{" "}
          <button className="underline" onClick={() => fileInput.current?.click()}>choose files</button>
          {" / "}
          <button className="underline" onClick={() => dirInput.current?.click()}>a folder</button>
        </p>
        <p className="mt-3 text-xs text-muted">{extensions.join("  ")}</p>
        <input ref={fileInput} type="file" multiple hidden accept={extensions.join(",")} onChange={(e) => accept(e.target.files)} />
        <input
          ref={dirInput}
          type="file"
          hidden
          {...({ webkitdirectory: "", directory: "" } as Record<string, string>)}
          onChange={(e) => accept(e.target.files)}
        />
      </div>

      {files.length > 0 && (
        <div className="rounded-lg border border-border bg-surface p-3 space-y-3">
          <ul className="max-h-40 overflow-auto text-sm divide-y divide-border">
            {files.map((f, i) => (
              <li key={i} className="flex justify-between gap-4 py-1">
                <span className="truncate">{f.webkitRelativePath || f.name}</span>
                <span className="flex gap-3 text-muted tabular-nums shrink-0">
                  {(f.size / 1024).toFixed(0)} KB
                  <button onClick={() => setFiles(files.filter((_, j) => j !== i))} aria-label="Remove">×</button>
                </span>
              </li>
            ))}
          </ul>
          <div className="flex gap-2">
            <input
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="Run name (optional)"
              className="flex-1 rounded border border-border bg-bg px-3 py-1.5 text-sm"
            />
            <button
              onClick={submit}
              disabled={busy}
              className="rounded bg-accent px-4 py-1.5 text-sm font-medium text-white dark:text-black disabled:opacity-50"
            >
              {busy ? "Uploading…" : `Run pipeline on ${files.length} file${files.length > 1 ? "s" : ""}`}
            </button>
          </div>
        </div>
      )}
      {error && <p className="text-sm text-bad">{error}</p>}
    </div>
  );
}
