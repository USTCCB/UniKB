# 3 步把 UniKB 推到你的 GitHub

## 步骤 1：在 GitHub 网页创建空仓库

打开 https://github.com/new
- Repository name: `UniKB`
- Description: `通用企业级 RAG 知识库平台 · Multi-Agent + MCP + Hybrid Search`
- Public / Private 随你（建议 Public，对求职有用）
- **不要**勾选 Add a README / Add .gitignore（本地已有）

## 步骤 2：本地初始化并提交

打开 PowerShell，cd 到 `E:\sys\UniKB`，执行：

```powershell
cd E:\sys\UniKB
git init
git add .
git commit -m "feat: initial UniKB - Multi-Agent RAG platform"
git branch -M main
git remote add origin https://github.com/USTCCB/UniKB.git
git push -u origin main
```

> 如果你的 GitHub 用户名不是 USTCCB，把上面两处的 `USTCCB` 换成你的用户名。

## 步骤 3：补 Secrets（可选但推荐）

到 GitHub 仓库 -> Settings -> Secrets and variables -> Actions：
- `DEEPSEEK_API_KEY`：你的 DeepSeek Key
- `OPENAI_API_KEY`：可选

然后 CI 就会自动跑 lint + smoke test + docker build。

## 常见问题

### 推送被拒绝（repository not empty）
GitHub 端已经初始化了 README。两种解法：
A. `git pull origin main --rebase` 后再 push
B. 在 GitHub 网页上 `git push -f` 强推（首次推空仓库时常见）

### 想换账号
```powershell
git config user.name  "你的名字"
git config user.email "你的邮箱"
```

### 想加 Star 数（求职技巧）
1. 写好 README 的截图/GIF（用 https://www.screencastify.com 或 terminalizer 录一段）
2. 发到 https://www.v2ex.com/ 的 AI 节点
3. 发到小红书 / 知乎的 AI 创作话题
4. 让同学/朋友帮忙 Star

### 面试展示建议
- 个人主页 README 顶部放一段 30s 演示 GIF
- 仓库加个 `docs/DEMO.md` 写 3 步本地跑起来的命令
- Issues 和 PRs 处理及时 -> 体现开源协作能力
