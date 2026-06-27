import Chat from "./Chat";

export default function Page() {
  return (
    <main className="container">
      <div className="header">
        <div className="brand">Uni<span>KB</span> · 企业级 RAG 知识库</div>
        <div className="muted">Multi-Agent · MCP · Hybrid Search</div>
      </div>
      <Chat />
    </main>
  );
}
