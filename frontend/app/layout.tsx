import "./globals.css";
import type { Metadata } from "next";
import Nav from "@/components/Nav";

export const metadata: Metadata = {
  title: "UniKB - 企业知识库",
  description: "Multi-Agent + MCP + Hybrid Search",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="zh-CN">
      <body>
        <Nav />
        <main className="container">{children}</main>
      </body>
    </html>
  );
}
