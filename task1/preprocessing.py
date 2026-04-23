import re

from bs4 import BeautifulSoup


def strip_html(text: str) -> str:
    if not text:
        return ""
    if "<" not in text or ">" not in text:
        return text
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text(separator="\n")


def normalize_text(text: str) -> str:
    text = strip_html(text or "")
    text = text.replace("\u3000", " ")
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def build_combined_text(title: str, content: str, max_chars: int) -> str:
    title = normalize_text(title)
    content = normalize_text(content)
    combined = f"标题：{title}\n正文：{content}".strip()
    if len(combined) > max_chars:
        return combined[:max_chars]
    return combined


def first_non_empty_sentence(text: str) -> str:
    if not text:
        return ""
    parts = re.split(r"[。！？!?；;\n]", text)
    for part in parts:
        part = part.strip()
        if len(part) >= 8:
            return part
    return text[:120].strip()
