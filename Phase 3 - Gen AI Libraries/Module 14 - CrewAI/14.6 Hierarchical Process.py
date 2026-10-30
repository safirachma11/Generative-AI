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

from crewai import Crew, Process, Agent, Task, LLM

if __name__ == "__main__":


    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    llm = LLM(
        model="openrouter/openrouter/free",
        api_key=api_key,
        temperature=0.1,
    )

    manager = Agent(
        role="AI Content Production Manager",
        goal="Coordinate a team to produce high-quality AI content efficiently.",
        backstory="""You are an experienced content production manager who excels at
breaking down projects, assigning work to the right specialist, and ensuring
quality standards are met before publication.""",
        llm=llm,
        allow_delegation=True,
        verbose=False,
    )
    researcher = Agent(
        role="AI Research Analyst",
        goal="Research AI topics thoroughly.",
        backstory="Senior researcher specialising in LLM systems and RAG.",
        llm=llm,
        verbose=False,
    )
    writer = Agent(
        role="Technical Writer",
        goal="Write developer-focused technical content.",
        backstory="Technical writer with Python and ML experience.",
        llm=llm,
        verbose=False,
    )

    main_task = Task(
        description="""Produce a complete technical brief on 'Vector Databases for RAG in Production'.
Include: what they are, how to choose one, and three implementation best practices.""",
        expected_output="A structured technical brief in Markdown, 400-500 words.",
        agent=manager,
    )

    crew = Crew(
        agents=[researcher, writer],
        tasks=[main_task],
        process=Process.hierarchical,
        manager_agent=manager,
        verbose=False,
    )

    print("Hierarchical Manager-led Crew compiled successfully.")
    print(f"- Manager Role: {manager.role}")
    print(f"- Managed Workers: {[researcher.role, writer.role]}")
    print(f"- Process Type: {crew.process}")
    print("Hierarchical Crew siap mengoordinasikan delegasi tugas!")

