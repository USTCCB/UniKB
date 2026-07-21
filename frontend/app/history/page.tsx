"use client";
import { useEffect, useState } from "react";
import {
  SessionDetail,
  deleteSession,
  getStoredToken,
  getStoredUser,
  getSession,
  listSessions,
} from "@/lib/api";

function fmtTime(ts: number): string {
  if (!ts) return "-";
  const d = new Date(ts * 1000);
  return d.toLocaleString("zh-CN", { hour12: false });
}

export default function HistoryPage() {
  const [token, setToken] = useState("");
  const [user, setUser] = useState("");
  const [sessions, setSessions] = useState<SessionDetail[] | null>(null);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [error, setError] = useState("");

  async function refresh() {
    if (!token) return;
    try {
      const list = await listSessions(token);
      const details: SessionDetail[] = await Promise.all(
        list.map((m) => getSession(token, m.session_id)),
      );
      setSessions(details);
      if (!activeId && details.length) setActiveId(details[0].session_id);
    } catch (e: any) {
      setError(e?.message ?? String(e));
    }
  }

  useEffect(() => {
    setToken(getStoredToken());
    setUser(getStoredUser());
  }, []);

  useEffect(() => {
    refresh();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [token]);

  async function onDelete(id: string) {
    if (!confirm("确认删除该会话？")) return;
    try {
      await deleteSession(token, id);
      setActiveId(null);
      refresh();
    } catch (e: any) {
      setError(e?.message ?? String(e));
    }
  }

  if (!token) {
    return (
      <div className="card">
        <div className="muted">请先登录。</div>
      </div>
    );
  }

  const active = sessions?.find((s) => s.session_id === activeId);

  return (
    <div className="history-page">
      <div className="row" style={{ justifyContent: "space-between", marginBottom: 12 }}>
        <h2 style={{ margin: 0 }}>历史会话</h2>
        <span className="muted">{user} · {sessions?.length ?? 0} 个会话</span>
      </div>

      {error && <div className="err">{error}</div>}

      <div className="hist-layout">
        <aside className="hist-list card">
          {sessions === null && <div className="muted">加载中...</div>}
          {sessions && sessions.length === 0 && <div className="muted">还没有会话，去 Chat 开始第一次提问吧</div>}
          {sessions?.map((s) => (
            <button
              key={s.session_id}
              className={"hist-item" + (s.session_id === activeId ? " active" : "")}
              onClick={() => setActiveId(s.session_id)}
            >
              <div className="title">{s.title}</div>
              <div className="meta muted">
                {s.message_count} 条 · {fmtTime(s.updated_at)}
              </div>
            </button>
          ))}
        </aside>

        <section className="hist-detail card">
          {!active && <div className="muted">选择一个会话查看</div>}
          {active && (
            <>
              <div className="row" style={{ justifyContent: "space-between" }}>
                <h3 style={{ margin: 0 }}>{active.title}</h3>
                <button className="ghost" onClick={() => onDelete(active.session_id)}>删除</button>
              </div>
              <div className="muted mono" style={{ fontSize: 11 }}>{active.session_id}</div>
              <div className="msgs">
                {active.messages.map((m, i) => (
                  <div key={i} className={"bubble " + m.role}>
                    <div className="role">{m.role === "user" ? "你" : "助手"}</div>
                    <div className="content">{m.content}</div>
                    {m.sources && m.sources.length > 0 && (
                      <details className="sources">
                        <summary>{m.sources.length} 条引用</summary>
                        {m.sources.map((s, j) => (
                          <div key={j} className="source">{s.content}</div>
                        ))}
                      </details>
                    )}
                  </div>
                ))}
                {active.messages.length === 0 && <div className="muted">(空会话)</div>}
              </div>
            </>
          )}
        </section>
      </div>
    </div>
  );
}
