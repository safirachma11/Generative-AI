import os
import sys
import pathlib
from dotenv import load_dotenv

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# Konfigurasi workspace cache & permissions CrewAI untuk Windows
cache_dir = pathlib.Path(__file__).parent / ".crewai_cache"
cache_dir.mkdir(parents=True, exist_ok=True)
os.environ["LOCALAPPDATA"] = str(cache_dir)
os.environ["CREWAI_TELEMETRY_OPT_OUT"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"

import crewai_core.settings
crewai_core.settings.DEFAULT_CONFIG_PATH = cache_dir / "settings.json"

load_dotenv()

from crewai import Task

def make_tasks(researcher, writer, reviewer):
    research_task = Task(
        description="""Research the current state of RAG (Retrieval-Augmented Generation) in 2024-2025.
Cover:
1. What RAG is and how it works
2. The main variants (naive RAG, advanced RAG, modular RAG)
3. Common production challenges and how practitioners solve them
4. Three real-world use cases with specific results
Be specific and factual. Avoid vague generalities.""",
        expected_output="""A structured research report with:
- A clear definition and architecture overview
- Descriptions of the three RAG variants with key differences
- Three specific production challenges with concrete solutions
- Three real-world use cases with measurable outcomes
Format: Markdown with headers.""",
        agent=researcher,
    )

    writing_task = Task(
        description="""Using the research report provided, write a developer guide titled
'Building Production RAG Systems'.
The guide should:
- Be written for senior developers with Python experience
- Include a conceptual overview, implementation considerations, and common pitfalls
- Use active voice and concrete examples
- Be 600-800 words""",
        expected_output="A complete, publication-ready developer guide in Markdown format.",
        agent=writer,
        context=[research_task],  # this task depends on research_task output
    )

    review_task = Task(
        description="""Review the developer guide for:
1. Technical accuracy - flag any claims that are incorrect or unsupported
2. Completeness - identify important topics that are missing
3. Clarity - suggest any sections that need rewriting
4. Audience fit - check that the level is appropriate for senior developers
Provide a structured review with specific, actionable feedback.""",
        expected_output="""A quality review report containing:
- Overall assessment (Ready / Needs Revision)
- Specific accuracy issues (if any) with corrections
- Missing content recommendations
- Clarity improvement suggestions
- Final verdict""",
        agent=reviewer,
        context=[writing_task],
    )
    return research_task, writing_task, review_task

if __name__ == "__main__":
    from crewai import Agent, LLM

    mock_llm = LLM(model="openrouter/openrouter/free", api_key="dummy")
    a1 = Agent(role="Researcher", goal="Research", backstory="Expert", llm=mock_llm)
    a2 = Agent(role="Writer", goal="Write", backstory="Expert", llm=mock_llm)
    a3 = Agent(role="Reviewer", goal="Review", backstory="Expert", llm=mock_llm)

    t1, t2, t3 = make_tasks(a1, a2, a3)
    print(f"- Task 1: {t1.description[:60]}...")
    print(f"- Task 2: {t2.description[:60]}... (Context: {[c.description[:20] for c in t2.context]})")
    print(f"- Task 3: {t3.description[:60]}... (Context: {[c.description[:20] for c in t3.context]})")
    print("Pabrik tugas (Task generator factory) berhasil dikompilasi dan siap dijalankan!")
