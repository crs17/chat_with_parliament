import os
from dataclasses import dataclass
from typing import Annotated

from dotenv import load_dotenv
import logfire
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext, ModelSettings
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider
import weaviate

from backend.common import PARTY_MANIFESTS, PARTY_NAMES, weaviate_collection_name
from backend.utils import create_retry_client

load_dotenv()

logfire.configure()
logfire.instrument_pydantic_ai()


# --- Data models ---
class PartyRecommendation(BaseModel):
    party_id: str
    score: Annotated[int, Field(strict=True, ge=-10, le=10)]
    reason: str


class PartyRecommendationList(BaseModel):
    """Structured output for the analyst agent."""

    results: list[PartyRecommendation]


@dataclass
class PartyExpertDeps:
    """Dependencies injected at call time to tell the party expert which party to query."""

    party_id: str


# --- Model setup ---
if os.getenv("ANTHROPIC_API_KEY"):
    model = AnthropicModel(
        os.getenv("ANTHROPIC_MODEL"),
        provider=AnthropicProvider(
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            http_client=create_retry_client()
        ),
        settings=ModelSettings(temperature=0.0, seed=42)
    )
else:
    ollama = OllamaProvider(
        base_url=os.getenv("OLLAMA_BASE_URL"),
        api_key=os.getenv("OLLAMA_API_KEY")
    )
    MODEL_NAME = os.getenv("OLLAMA_MODEL")
    model = OpenAIChatModel(
        MODEL_NAME, provider=ollama,
        settings=ModelSettings(temperature=0.0, seed=42)
    )


# --- Party Expert Agent ---

party_expert_agent = Agent[PartyExpertDeps, str](
    model=model,
    instructions=(
        "Du er en ekspert i danske partiprogrammer. "
        "Du får et spørgsmål om et specifikt parti, som du skal svare på. "
        "Brug get_context til at finde relevante informationer fra partiets partiprogram. "
        "get_context kan bruges med string queries. Brug IKKE JSON."
        "Brug kun informationer direkte fra partiprogrammet. "
        "Hvis get_context returnerer 0 chunks, så svar 'Ingen relevant information om {query} i partiprogrammet.' "
        "Tilføj ikke fakta, som ikke fremgår direkte fra partiprogrammet. "
        "Hold dit svar kort og præcist (maks. 3-4 sætninger). "
        "Svar altid på dansk."
    ),
    output_type=str,
    retries=2,   
)


@party_expert_agent.tool
async def get_context(ctx: RunContext[PartyExpertDeps], query: str) -> list[str]:
    """Søg i partiets partiprogram efter relevante informationer."""
    party_id = ctx.deps.party_id
    party_name = PARTY_NAMES.get(party_id, party_id)
    print(f"  [RAG] {party_name} ({party_id}): '{query}'")
    with weaviate.connect_to_local() as client:
        collection = client.collections.get(weaviate_collection_name(party_id))
        response = collection.query.near_text(query=query, limit=2)
    texts = [item.properties["text"] for item in response.objects]
    print(f"  [RAG] Found {len(texts)} chunks for {party_name}")
    return texts


# --- Political Analyst Agent ---

party_list_str = "\n".join(
    f"- {pid}: {PARTY_NAMES[pid]}" for pid in PARTY_MANIFESTS
)

political_analyst_prompt = f"""
\no_think
Du er en skarp og kritisk politisk analytiker. Du arbejder udelukkende på dansk.

Du har adgang til en parti-ekspert via consult_party_expert.
Kald consult_party_expert med et party_id og en kort query på dansk for hvert parti.

Baseret på brugerens input:
1. Formulér en relevant query (fx "klimapolitik", "sundhedsvæsenet", "skattepolitik").
2. Kald consult_party_expert for hvert parti.
3. Vurdér hvert parti kritisk baseret på ekspertens svar.
4. Giv en score fra -10 (værste match) til 10 (bedste match) med begrundelse.

Medmindre brugeren spørger om specifikke partier, skal ALLE partier vurderes.

Gyldige party_id'er:
{party_list_str}
"""

political_analyst_agent = Agent(
    model=model,
    instructions=political_analyst_prompt,
    output_type=PartyRecommendationList,
    retries=3,
)


@political_analyst_agent.tool
async def consult_party_expert(ctx: RunContext, party_id: str, query: str) -> str:
    """Konsultér parti-eksperten om et specifikt parti.

    Args:
        party_id: Partiets ID (fx 'A', 'B', 'V', 'Ø').
        query: Spørgsmål på dansk (fx 'klimapolitik', 'boligpolitik').

    Returns:
        Ekspertens svar baseret på partiets partiprogram.
    """
    if party_id not in PARTY_MANIFESTS:
        return f"Ukendt parti: {party_id}. Gyldige: {', '.join(PARTY_MANIFESTS.keys())}"

    party_name = PARTY_NAMES[party_id]
    print(f"\n>>> Analyst → Party Expert: {party_name} ({party_id}), query='{query}'")

    deps = PartyExpertDeps(party_id=party_id)
    result = await party_expert_agent.run(
        f"Hvad siger {party_name} om: {query}",
        deps=deps,
    )

    print(f"<<< Party Expert → Analyst: {party_name} done ({len(result.output)} chars)\n")
    return result.output
