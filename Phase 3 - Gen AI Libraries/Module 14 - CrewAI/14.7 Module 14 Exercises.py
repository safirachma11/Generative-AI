import os
import sys
import pathlib
from typing import Type
from pydantic import BaseModel, Field

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

from crewai.tools import BaseTool

class MockSearchInput(BaseModel):
    query: str = Field(description="Search query string")

class MockWebSearchTool(BaseTool):
    name: str = "Mock Web Search Tool"
    description: str = "Search the web for technical documentation and guides."
    args_schema: Type[BaseModel] = MockSearchInput

    def _run(self, query: str) -> str:
        return f"Results for '{query}': RAG connects LLMs to external vector databases like ChromaDB."

if __name__ == "__main__":
    print("=" * 60)
    print("14.7 CrewAI Exercises - Custom Tool Implementation")
    print("=" * 60)
    tool = MockWebSearchTool()
    query = "fine-tuning vs RAG"
    result = tool._run(query)
    print(f"Tool Name   : {tool.name}")
    print(f"Description : {tool.description}")
    print(f"Input Query : '{query}'")
    print(f"Tool Output : {result}")
