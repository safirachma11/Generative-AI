import os
import sys
import pathlib
import importlib.util
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

from crewai import Crew, Process, Agent, LLM

def _import_module(module_name: str, filename: str):
    file_path = pathlib.Path(__file__).parent / filename
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

if __name__ == "__main__":
   

    # 1. Import tools dari 14.3 Tools.py
    tools_mod = _import_module("tools_mod", "14.3 Tools.py")
    cost_tool = tools_mod.cost_tool
    context_tool = tools_mod.context_tool

    # 2. Import task generator dari 14.4 Tasks.py
    tasks_mod = _import_module("tasks_mod", "14.4 Tasks.py")
    make_tasks = tasks_mod.make_tasks

    # 3. Inisialisasi LLM CrewAI (Mendukung OpenRouter free tier)
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    llm = LLM(
        model="openrouter/openrouter/free",
        api_key=api_key,
        temperature=0.1,
    )

    # 4. Inisialisasi 3 Agent spesialis
    researcher = Agent(
        role="AI Research Analyst",
        goal="Find accurate information on AI topics.",
        backstory="Senior AI research analyst with expertise in LLM systems.",
        llm=llm,
        tools=[context_tool],
        verbose=False,
        max_iter=3,
    )

    writer = Agent(
        role="Technical Content Writer",
        goal="Write clear technical content for developers.",
        backstory="Technical writer with a software engineering background.",
        llm=llm,
        tools=[cost_tool],
        verbose=False,
        max_iter=3,
    )

    reviewer = Agent(
        role="Content Quality Reviewer",
        goal="Ensure content is accurate and publication-ready.",
        backstory="Meticulous editor who reviews technical AI content.",
        llm=llm,
        verbose=False,
        max_iter=3,
    )

    # 5. Bangun dependensi tugas
    research_task, writing_task, review_task = make_tasks(researcher, writer, reviewer)

    # 6. Rakit Sequential Crew
    crew = Crew(
        agents=[researcher, writer, reviewer],
        tasks=[research_task, writing_task, review_task],
        process=Process.sequential,
        verbose=False,
        memory=False,
    )

    print("Sequential Crew assembled successfully.")
    print(f"- Total Agents: {len(crew.agents)}")
    print(f"- Total Tasks : {len(crew.tasks)}")
    print(f"- Process Type: {crew.process}")
    print("Crew siap dijalankan dengan crew.kickoff()!")

