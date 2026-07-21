"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import AuthBar from "./AuthBar";

const TABS = [
  { href: "/chat", label: "Chat" },
  { href: "/upload", label: "Upload" },
  { href: "/sources", label: "Sources" },
  { href: "/history", label: "History" },
];

export default function Nav() {
  const path = usePathname();
  return (
    <header className="topbar">
      <div className="brand">
        Uni<span>KB</span>
        <small className="muted">· 企业级 RAG 知识库</small>
      </div>
      <nav className="tabs">
        {TABS.map((t) => (
          <Link
            key={t.href}
            href={t.href}
            className={"tab" + (path?.startsWith(t.href) ? " active" : "")}
          >
            {t.label}
          </Link>
        ))}
      </nav>
      <AuthBar />
    </header>
  );
}
