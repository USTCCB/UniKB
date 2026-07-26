"use client";
import { useEffect, useState } from "react";
import { clearStoredAuth, devToken, login, register, setStoredToken, getStoredToken, getStoredUser } from "@/lib/api";

export default function AuthBar() {
  const [token, setToken] = useState("");
  const [user, setUser] = useState("");
  const [mode, setMode] = useState<"dev" | "login" | "register">("dev");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [email, setEmail] = useState("");
  const [err, setErr] = useState("");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const sync = () => {
      setToken(getStoredToken());
      setUser(getStoredUser());
    };
    sync();
    window.addEventListener("unikb:auth", sync);
    return () => window.removeEventListener("unikb:auth", sync);
  }, []);

  async function submit() {
    setErr("");
    setLoading(true);
    try {
      let u = username;
      let resp;
      if (mode === "dev") {
        resp = await devToken();
        u = "dev-user";
      } else if (mode === "login") {
        resp = await login(username, password);
        u = username;
      } else {
        resp = await register(username, email, password);
        u = username;
      }
      setStoredToken(resp.access_token, u);
      setPassword("");
    } catch (e: any) {
      setErr(e?.message || String(e));
    } finally {
      setLoading(false);
    }
  }

  function logout() {
    clearStoredAuth();
  }

  if (token) {
    return (
      <div className="authbar">
        <span className="muted">当前用户</span>
        <strong>{user}</strong>
        <button className="ghost" onClick={logout}>退出</button>
      </div>
    );
  }

  return (
    <div className="authbar">
      <select value={mode} onChange={(e) => setMode(e.target.value as any)}>
        <option value="dev">Dev 一键 Token</option>
        <option value="login">登录</option>
        <option value="register">注册</option>
      </select>
      {mode !== "dev" && (
        <input
          type="text"
          placeholder="用户名"
          value={username}
          onChange={(e) => setUsername(e.target.value)}
        />
      )}
      {mode !== "dev" && (
        <input
          type="password"
          placeholder="密码"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
        />
      )}
      {mode === "register" && (
        <input
          type="email"
          placeholder="邮箱"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
        />
      )}
      <button onClick={submit} disabled={loading}>{loading ? "..." : "进入"}</button>
      {err && <span className="err">{err}</span>}
    </div>
  );
}
