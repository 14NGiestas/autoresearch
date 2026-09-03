from .base import BasePlatform
from .claude import Claude
from .chatgpt import ChatGPT
from .grok import Grok
from .qwen import Qwen
from .deepseek import Deepseek
from .gemini import Gemini

PLATFORMS: dict[str, BasePlatform] = {
    p.name: p
    for p in (Claude(), ChatGPT(), Grok(), Qwen(), Deepseek(), Gemini())
}

ALL_PATTERNS = {name: p.pattern for name, p in PLATFORMS.items()}
