import "./globals.css";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "UniKB - 企业知识库",
  description: "Multi-Agent + MCP + Hybrid Search",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="zh-CN">
      <body>{children}</body>
    </html>
  );
}
