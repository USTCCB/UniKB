"use client";
import { useEffect, useState } from "react";
import { getStoredToken, listDocuments, uploadDocument } from "@/lib/api";

type DocRow = {
  filename: string;
  chunks: number;
  status: string;
  doc_id: string;
  message?: string;
};

export default function UploadPage() {
  const [token, setToken] = useState("");
  const [files, setFiles] = useState<File[]>([]);
  const [busy, setBusy] = useState(false);
  const [rows, setRows] = useState<DocRow[]>([]);
  const [stats, setStats] = useState<{ bm25_count: number; vector_count: number } | null>(null);
  const [err, setErr] = useState("");

  useEffect(() => {
    setToken(getStoredToken());
  }, []);

  async function refresh() {
    if (!token) return;
    try {
      const s = await listDocuments(token);
      setStats(s);
    } catch (e: any) {
      setErr(e?.message ?? String(e));
    }
  }

  useEffect(() => {
    refresh();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [token]);

  async function submit() {
    if (!files.length || !token) return;
    setBusy(true);
    setErr("");
    const next: DocRow[] = [];
    for (const f of files) {
      try {
        const r = await uploadDocument(token, f);
        next.push({ filename: r.filename, chunks: r.chunks, status: r.status, doc_id: r.doc_id, message: r.message });
      } catch (e: any) {
        next.push({ filename: f.name, chunks: 0, status: "error", doc_id: "", message: e?.message ?? String(e) });
      }
    }
    setRows((r) => [...next, ...r]);
    setFiles([]);
    setBusy(false);
    refresh();
  }

  function drop(e: React.DragEvent<HTMLDivElement>) {
    e.preventDefault();
    const list = Array.from(e.dataTransfer.files || []);
    setFiles((prev) => [...prev, ...list]);
  }

  if (!token) {
    return (
      <div className="card">
        <div className="muted">请先登录。</div>
      </div>
    );
  }

  return (
    <div className="upload-page">
      <div className="card">
        <h3>拖拽文档到下方区域</h3>
        <div
          className="dropzone"
          onDragOver={(e) => e.preventDefault()}
          onDrop={drop}
          onClick={() => document.getElementById("file-input")?.click()}
        >
          <input
            id="file-input"
            type="file"
            multiple
            hidden
            onChange={(e) => setFiles(Array.from(e.target.files || []))}
          />
          <span className="muted">支持 PDF / Markdown / DOCX / TXT / 图片</span>
        </div>
        {files.length > 0 && (
          <ul className="filelist">
            {files.map((f, i) => (
              <li key={i}>
                {f.name} <span className="muted">{(f.size / 1024).toFixed(1)} KB</span>
              </li>
            ))}
          </ul>
        )}
        <div className="row" style={{ marginTop: 12 }}>
          <button onClick={submit} disabled={busy || files.length === 0}>
            {busy ? "上传中..." : `上传 ${files.length} 个文件`}
          </button>
          {stats && (
            <span className="muted">
              当前知识库: BM25 {stats.bm25_count} chunks · 向量 {stats.vector_count} chunks
            </span>
          )}
        </div>
        {err && <div className="err">{err}</div>}
      </div>

      {rows.length > 0 && (
        <div className="card">
          <h3>本次上传结果</h3>
          <table className="tbl">
            <thead>
              <tr>
                <th>文件名</th>
                <th>chunks</th>
                <th>状态</th>
                <th>doc_id</th>
                <th>消息</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r, i) => (
                <tr key={i} className={r.status === "error" ? "err-row" : ""}>
                  <td>{r.filename}</td>
                  <td>{r.chunks}</td>
                  <td>{r.status}</td>
                  <td className="mono">{r.doc_id || "-"}</td>
                  <td className="muted">{r.message}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
