import type { JobDetail } from "@/lib/api";
import { LANG_NAMES } from "@/lib/api";
import { Bars, Card, Stat } from "./ui";

export function Overview({ job, onOpenDoc }: { job: JobDetail; onOpenDoc: (docId: string) => void }) {
  const s = job.summary;
  const total = job.n_sentences;
  const usable = s.usable["1"] ?? 0;
  const dup = s.duplicate["1"] ?? 0;
  const langs = Object.fromEntries(Object.entries(s.lang).map(([k, v]) => [LANG_NAMES[k] ?? k, v]));
  const annotated = Object.values(s.annotated).reduce((a, b) => a + b, 0);

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <Stat label="Documents" value={job.n_docs} sub={`${job.n_files} uploaded`} />
        <Stat label="Sentences" value={total.toLocaleString()} />
        <Stat label="Usable" value={total ? `${Math.round((usable / total) * 100)}%` : "—"} sub={`${dup} duplicates`} />
        <Stat label="Human labels" value={annotated} sub="across all tasks" />
      </div>

      {job.errors.length > 0 && (
        <Card title={<span className="text-bad">Extraction errors</span>}>
          <ul className="text-sm space-y-1">
            {job.errors.map((e, i) => (
              <li key={i}><span className="font-medium">{e.file ?? "run"}</span>: <span className="text-muted">{e.error}</span></li>
            ))}
          </ul>
        </Card>
      )}

      <div className="grid md:grid-cols-2 gap-4">
        <Card title="Language"><Bars data={langs} /></Card>
        <Card title="Agri vs non-agri (predicted)"><Bars data={s.predicted.agri ?? {}} /></Card>
        <Card title="Agri sub-domain (predicted)">
          <Bars data={Object.fromEntries(Object.entries(s.predicted.agri_subdomain ?? {}).filter(([k]) => k !== "not_applicable"))} />
        </Card>
        <Card title="Source types"><Bars data={s.source_type} /></Card>
      </div>

      <Card title="Documents">
        <div className="overflow-x-auto -m-4">
          <table className="w-full text-sm">
            <thead className="text-left text-muted border-b border-border">
              <tr>
                <th className="px-4 py-2 font-medium">File</th>
                <th className="px-4 py-2 font-medium">Type</th>
                <th className="px-4 py-2 font-medium text-right">Sentences</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border">
              {job.documents.map((d) => (
                <tr key={d.doc_id} className="hover:bg-surface-2 cursor-pointer" onClick={() => onOpenDoc(d.doc_id)}>
                  <td className="px-4 py-2">{d.source_path}</td>
                  <td className="px-4 py-2 text-muted">{d.source_type}</td>
                  <td className={`px-4 py-2 text-right tabular-nums ${d.n_sentences === 0 ? "text-bad" : ""}`}>
                    {d.n_sentences}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}
