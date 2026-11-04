export type Job = {
  job_id: string;
  name: string;
  status: "queued" | "running" | "done" | "failed";
  created_at: number;
  finished_at: number | null;
  n_files: number;
  n_docs: number;
  n_sentences: number;
  errors: { file: string | null; error: string }[];
};

export type DocumentRow = {
  doc_id: string;
  source_path: string;
  source_type: string;
  n_sentences: number;
};

export type Summary = {
  lang: Record<string, number>;
  usable: Record<string, number>;
  duplicate: Record<string, number>;
  source_type: Record<string, number>;
  predicted: Record<string, Record<string, number>>;
  annotated: Record<string, number>;
};

export type JobDetail = Job & { summary: Summary; documents: DocumentRow[] };

export type Prediction = {
  label: string;
  score: number | null;
  classifier: string;
  matched?: Record<string, number>;
};

export type Sentence = {
  sent_id: string;
  doc_id: string;
  index: number;
  text: string;
  source_path: string;
  source_type: string;
  location: string;
  lang: string;
  script: string;
  quality: {
    usable: boolean;
    duplicate: boolean;
    flags: string[];
    alpha_ratio: number;
    garbled_ratio: number;
    n_words: number;
  };
  labels: Record<string, Prediction>;
  annotations: Record<string, { label: string; note: string; annotator: string }>;
};

export type Meta = {
  extensions: string[];
  tasks: Record<string, string[]>;
  classifiers: { name: string; task: string }[];
};

export type Report = {
  n: number;
  accuracy: number | null;
  macro_f1: number | null;
  labels: string[];
  per_class: Record<string, { precision: number; recall: number; f1: number; support: number }>;
  confusion: Record<string, Record<string, number>>;
};

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`/api${path}`, { cache: "no-store", ...init });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      detail = (await res.json()).detail ?? detail;
    } catch {}
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return res.json();
}

export const api = {
  meta: () => req<Meta>("/meta"),
  jobs: () => req<Job[]>("/jobs"),
  job: (id: string) => req<JobDetail>(`/jobs/${id}`),
  deleteJob: (id: string) => req(`/jobs/${id}`, { method: "DELETE" }),
  createJob: (files: File[], name = "") => {
    const fd = new FormData();
    // webkitRelativePath preserves folder structure for directory uploads.
    for (const f of files) fd.append("files", f, f.webkitRelativePath || f.name);
    fd.append("name", name);
    return req<{ job_id: string; accepted: number; skipped: string[] }>("/jobs", {
      method: "POST",
      body: fd,
    });
  },
  sentences: (id: string, params: Record<string, string | number | boolean | undefined>) => {
    const qs = new URLSearchParams();
    for (const [k, v] of Object.entries(params)) if (v !== undefined && v !== "") qs.set(k, String(v));
    return req<{ total: number; page: number; page_size: number; items: Sentence[] }>(
      `/jobs/${id}/sentences?${qs}`,
    );
  },
  annotate: (sentId: string, task: string, label: string | null, note = "") =>
    req(`/sentences/${encodeURIComponent(sentId)}/annotations`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ task, label, note }),
    }),
  metrics: (id: string) => req<Record<string, Report>>(`/jobs/${id}/metrics`),
  exportUrl: (id: string) => `/api/jobs/${id}/export`,
};

export const LANG_NAMES: Record<string, string> = {
  mr: "Marathi",
  hi: "Hindi",
  en: "English",
  "und-Deva": "Devanagari (unsure)",
  und: "Unknown",
};

export const pretty = (s: string) => s.replace(/_/g, " ");
