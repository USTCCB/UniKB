"""文档清洗 (Document Cleaning).

RAG 质量调优里有句经验: **清洗占 60%~70% 的精力**, 检索/重排只占小头.
解析出的原始文本充满噪声 (控制字符、零宽字符、页眉页脚、脚注、OCR 残片、
Markdown/LaTeX 残留等), 直接切块会污染向量、拉低召回。

这里用 **42 条正则规则** 做确定性清洗 (不依赖 LLM, 可复现、零成本):
  1. 不可见字符: NUL/控制字符、零宽空格/连字、BOM、软连字符
  2. 排版噪声: 多余空白/换行、行尾空格、Tab 混用
  3. 标记语言残留: HTML 标签、Markdown 链接/图片/加粗、LaTeX 命令与数学环境
  4. 结构性噪声: 页眉页脚 (`第 N 页`)、脚注标记 (`¹`/`[1]`)、目录点线 (`.....`)
  5. 半结构化噪声: 连续分隔线、孤立标点、全角空格、URL/邮箱归一

清洗是**幂等**的: 对同一文本多次 clean 结果一致。
"""
from __future__ import annotations

import re
from typing import List

# ---------------------------------------------------------------------------
# 1. 不可见 / 控制字符
# ---------------------------------------------------------------------------
# NUL 与 C0/C1 控制字符 (保留 \t \n \r 三个排版字符)
_RE_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
# 零宽字符: 零宽空格、零宽连字、词连接符、BOM、行/段落分隔符
_RE_ZERO_WIDTH = re.compile(
    "[\u200b\u200c\u200d\u2060\ufeff\u202a-\u202e\u00ad\ufffc]"
)
# 特殊的 "软连字符" 与 "词内连接" 已是上面一部分, 这里补零宽空格变体
_RE_ZWSP_VARIANT = re.compile("[\u200b\u200e\u200f]")

# ---------------------------------------------------------------------------
# 2. HTML / XML 残留
# ---------------------------------------------------------------------------
# 注意 negative lookahead: 排除清洗器自己产出的 <URL> / <EMAIL> 占位符.
# 否则对同一文本 clean 两次, 第二遍会把占位符当成 HTML 标签抹掉, 破坏幂等性.
_RE_HTML_TAG = re.compile(r"<(?!URL>|EMAIL>)[^>]+>")
_RE_HTML_ENTITY = re.compile(r"&(?:[a-zA-Z]+|#\d+);")
_RE_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_RE_CDATA = re.compile(r"<!\[CDATA\[.*?\]\]>", re.DOTALL)

# ---------------------------------------------------------------------------
# 3. Markdown / 富文本残留
# ---------------------------------------------------------------------------
_RE_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")  # [文本](url) -> 文本
_RE_MD_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")  # ![alt](url) -> 删
_RE_MD_BOLD_ITALIC = re.compile(r"[*_]{1,3}([^*_]+)[*_]{1,3}")  # **x** -> x
_RE_MD_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+", re.MULTILINE)  # 标题 # 去掉
_RE_MD_BLOCKQUOTE = re.compile(r"^\s{0,3}>\s?", re.MULTILINE)
_RE_MD_HR = re.compile(r"^\s{0,3}[-*_]{3,}\s*$", re.MULTILINE)
_RE_MD_CODE_FENCE = re.compile(r"```.*?```", re.DOTALL)
_RE_MD_INLINE_CODE = re.compile(r"`([^`]+)`")
_RE_MD_TABLE_SEP = re.compile(r"^\s*\|?[\s:|-]+\|?\s*$", re.MULTILINE)  # |---|---|
_RE_MD_TABLE_PIPE = re.compile(r"(?<!\S)\|(?!\S)")  # 孤立的表格竖线

# ---------------------------------------------------------------------------
# 4. LaTeX 残留
# ---------------------------------------------------------------------------
_RE_LATEX_ENV = re.compile(r"\\begin\{[^}]*\}.*?\\end\{[^}]*\}", re.DOTALL)
_RE_LATEX_CMD = re.compile(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?(?:\{[^}]*\})?")
_RE_LATEX_MATH = re.compile(r"\$[^$]*\$")  # 行内数学
_RE_LATEX_DISPLAY = re.compile(r"\$\$.*?\$\$", re.DOTALL)  # 块级数学
_RE_LATEX_COMMENT = re.compile(r"%.*$", re.MULTILINE)

# ---------------------------------------------------------------------------
# 5. 结构性噪声: 页眉页脚 / 脚注 / 目录
# ---------------------------------------------------------------------------
_RE_PAGE_NUM = re.compile(r"^\s*(?:第\s*\d+\s*页\s*(?:[/-]\s*\d+\s*页)?|page\s*\d+\s*(?:of\s*\d+)?)\s*$", re.IGNORECASE | re.MULTILINE)
_RE_FOOTNOTE_MARK = re.compile(r"(?<=\S)[¹²³⁴⁵⁶⁷⁸⁹⁰⑩]")  # 上标脚注
_RE_FOOTNOTE_BRACKET = re.compile(r"\[\d{1,3}\](?=\s|$)")  # [12] 脚注
_RE_TOC_DOTS = re.compile(r"\.{4,}\s*\d+\s*$")  # 目录点线 `.....12`
_RE_SECTION_NUM = re.compile(r"^\s*\d+(?:\.\d+)*\s+(?=\S)")  # 章节号 `1.2.3 `

# ---------------------------------------------------------------------------
# 6. 标点 / 空白归一
# ---------------------------------------------------------------------------
_RE_TRAILING_SPACE = re.compile(r"[ \t]+$")  # 行尾空格
_RE_MULTI_SPACE = re.compile(r"[ \t]{2,}")  # 连续空格 -> 单空格
_RE_MULTI_NEWLINE = re.compile(r"\n{3,}")  # 连续空行
_RE_LEADING_SPACE_LINE = re.compile(r"^[ \t]+", re.MULTILINE)
_RE_FULLWIDTH_SPACE = re.compile(r"[\u3000]")  # 全角空格 -> 半角
_RE_MULTI_DOT = re.compile(r"\.{4,}")  # 多余点
_RE_ISOLATED_PUNCT = re.compile(r"(?<=\n)[，。、；：！？,.!?;:]+\s*$")  # 孤立标点行尾

# URL / 邮箱归一 (避免同一链接多种写法污染 chunk)
_RE_URL = re.compile(r"https?://[^\s<>\"']+|www\.[^\s<>\"']+", re.IGNORECASE)
_RE_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")

# 连续分隔线 (--- / === / ***)
_RE_SEPARATOR_LINE = re.compile(r"^\s*[-=*_]{5,}\s*$", re.MULTILINE)

# 空括号 / 空引用
_RE_EMPTY_PAREN = re.compile(r"\(\s*\)|\（\s*\）")
_RE_BROKEN_FRAGMENT = re.compile(r"[•·▪◦]+\s*$")  # 行尾孤立项目符号

# 中文标点前的空格 (HTML 标签被替换成空格后很容易残留 "文字 ，" 这种)
_RE_SPACE_BEFORE_CJK_PUNCT = re.compile(r"[ \t]+(?=[，。、；：！？）】》」』])")
_RE_TAB = re.compile(r"\t+")


class DocumentCleaner:
    """确定性文档清洗器, 42 条正则规则, 幂等、无 LLM 依赖。

    用法:
        text = DocumentCleaner().clean(raw_text)
    """

    # 顺序很重要: 先去标记语言/结构噪声, 再做空白归一, 最后兜底。
    _STEPS: List[tuple] = [
        # ---- 标记语言 / 结构 ----
        (_RE_HTML_COMMENT, ""),
        (_RE_CDATA, ""),
        (_RE_HTML_TAG, " "),
        (_RE_HTML_ENTITY, " "),
        (_RE_MD_CODE_FENCE, " "),
        (_RE_MD_IMAGE, ""),
        (_RE_MD_LINK, r"\1"),
        (_RE_MD_INLINE_CODE, r"\1"),
        (_RE_MD_BOLD_ITALIC, r"\1"),
        (_RE_MD_TABLE_SEP, ""),
        (_RE_MD_TABLE_PIPE, " "),
        (_RE_MD_HEADING, ""),
        (_RE_MD_BLOCKQUOTE, ""),
        (_RE_MD_HR, ""),
        (_RE_LATEX_DISPLAY, " "),
        (_RE_LATEX_MATH, " "),
        (_RE_LATEX_ENV, " "),
        (_RE_LATEX_COMMENT, ""),
        (_RE_LATEX_CMD, " "),
        # ---- 控制 / 不可见字符 ----
        (_RE_CONTROL, ""),
        (_RE_ZERO_WIDTH, ""),
        (_RE_ZWSP_VARIANT, ""),
        (_RE_TAB, " "),
        # ---- 结构性噪声 ----
        (_RE_PAGE_NUM, ""),
        (_RE_FOOTNOTE_MARK, ""),
        (_RE_FOOTNOTE_BRACKET, ""),
        (_RE_TOC_DOTS, ""),
        (_RE_SEPARATOR_LINE, ""),
        (_RE_BROKEN_FRAGMENT, ""),
        (_RE_EMPTY_PAREN, ""),
        # ---- 标点 / 空白归一 ----
        (_RE_FULLWIDTH_SPACE, " "),
        (_RE_URL, " <URL> "),
        (_RE_EMAIL, " <EMAIL> "),
        (_RE_TRAILING_SPACE, ""),
        (_RE_LEADING_SPACE_LINE, ""),
        (_RE_ISOLATED_PUNCT, ""),
        (_RE_MULTI_DOT, "..."),
        (_RE_MULTI_SPACE, " "),
        (_RE_SPACE_BEFORE_CJK_PUNCT, ""),
        (_RE_MULTI_NEWLINE, "\n\n"),
    ]

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def clean(self, text: str) -> str:
        if not self.enabled or not text:
            return (text or "").strip()
        out = text
        for pattern, repl in self._STEPS:
            out = pattern.sub(repl, out)
        # 收尾: 去掉章节号前缀里残留的多余空格, 折叠首尾空白
        out = _RE_SECTION_NUM.sub(lambda m: m.group(0).rstrip() + " ", out)
        out = re.sub(r"\n[ \t]+\n", "\n\n", out)
        return out.strip()

    # 暴露规则计数, 方便 README / 测试断言 "42 条"。
    @property
    def rule_count(self) -> int:
        return len(self._STEPS) + 2  # +2: 收尾的章节号替换与换行折叠
