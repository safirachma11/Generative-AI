import os
import sys
import pathlib
from dotenv import load_dotenv

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# 1. Konfigurasi direktori cache dan kredensial aman di Windows
cache_dir = pathlib.Path(__file__).parent / ".crewai_cache"
cache_dir.mkdir(parents=True, exist_ok=True)
os.environ["LOCALAPPDATA"] = str(cache_dir)
os.environ["CREWAI_TELEMETRY_OPT_OUT"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"

import crewai_core.settings
crewai_core.settings.DEFAULT_CONFIG_PATH = cache_dir / "settings.json"

load_dotenv()

from crewai import Agent, LLM

# 2. Inisialisasi LLM CrewAI (Mendukung OpenRouter free tier)
llm = LLM(
    model="openrouter/openrouter/free",
    api_key=os.environ.get("OPENROUTER_API_KEY", ""),
    temperature=0.1,
)

# 3. Definisi 3 Agent dengan peran, tujuan, dan backstory spesifik
researcher = Agent(
    role="AI Research Analyst",
    goal="Find accurate and up-to-date information on AI topics.",
    backstory="""You are a senior AI research analyst with 10 years of experience
reading and synthesising academic papers, technical blogs, and release notes.
You are known for being precise, citing sources, and never making things up.""",
    llm=llm,
    verbose=True,
    allow_delegation=False,
    max_iter=5,
)

writer = Agent(
    role="Technical Content Writer",
    goal="Write clear, accurate, and engaging technical content for developers.",
    backstory="""You are a technical writer with a background in software engineering.
You excel at turning complex AI concepts into practical, well-structured guides.
You always write in active voice, use concrete examples, and avoid jargon.""",
    llm=llm,
    verbose=True,
    allow_delegation=False,
)

reviewer = Agent(
    role="Content Quality Reviewer",
    goal="Ensure content is accurate, complete, and publication-ready.",
    backstory="""You are a meticulous editor who reviews technical content for
accuracy, clarity, and completeness. You check facts, flag unsupported claims,
and ensure the writing matches the target audience's level.""",
    llm=llm,
    verbose=True,
    allow_delegation=True,
)

if __name__ == "__main__":
    print(f"- Agent 1: {researcher.role}")
    print(f"- Agent 2: {writer.role}")
    print(f"- Agent 3: {reviewer.role}")
    print("Semua Agent CrewAI berhasil dikonfigurasi dan siap digunakan!")

