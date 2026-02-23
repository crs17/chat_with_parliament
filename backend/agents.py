import os
from dotenv import load_dotenv
from pydantic import BaseModel, conint
from pydantic_ai import Agent, RunContext, Tool
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider
import weaviate

from backend.common import PARTIPROGRAM_PATHS, PARTY_NAMES

load_dotenv()


class PartyRecommendation(BaseModel):
    party_id: str
    score: conint(strict=True, ge=-10, le=10)
    reason: str


ollama = OllamaProvider(base_url=os.getenv("OLLAMA_BASE_URL"))
model = OpenAIChatModel("qwen3:8b", provider=ollama)


party_expert_tools = []

for party_id in PARTIPROGRAM_PATHS:
    party_expert_agent = Agent[None, str](
        model=model,
        instructions=(
            f"Du er ekspert i partiet {PARTY_NAMES[party_id]} ({party_id}). "
            "Du får spørgsmål om partiet, som du skal svare på. "
            "Brug get_context tool til at finde relevante informationer om partiets partiprogram. "
            "Brug kun informationer, som er direkte fra partiets partiprogram. "
            "Tilføj ikke fakta, som ikke fremgår direkte fra partiets partiprogram. "
        ),
        output_type=str
    )

    @party_expert_agent.tool
    async def get_context(ctx: RunContext, query: str) -> list[str]:

        with weaviate.connect_to_local() as client:
            party_collection = client.collections.get(party_id)
            response = party_collection.query.near_text(
                query=query,
                limit=2,
            )

        return [item.properties['text'] for item in response.objects]


    party_expert_tools.append(Tool(
        get_context,
        name=f"get_party_context_{party_id}",
        takes_ctx=True,
    ))


political_analyst_agent_prompt = (
    """
Du er en skarp og kritisk politisk analytiker. 
Du får input fra brugerne om deres politiske overbevisning og hvad der er vigtigt for dem. 
Du kan bruge parti eksperterne til at finde informationer om hvert parti. 
Du skal kun undersøge de partier, som du har politiske eksperter for. 
Brug kun den information, som du får fra parti eksperterne. 
Du kan gå ud fra at partierne ønsker at appellere til så mange brugere som muligt. 
Det er derfor din opgave at vurderere partierne kritisk for at afgøre hvor godt de matcher brugerens ønkser. 
Du skal give en rating på hvor godt partiet matcher brugeren. Ratingen skal være mellem -10 (værste match med brugerens input) og 10 (bedste match med brugerens input). 
Du skal give en begrundelse for din rating. 
Du skal give en rating på hvor godt partiet kan håndtere det, som brugeren har bedt om. 

Du kan undersøge følgende partier:
""" + 
'\n'.join([f"{party_id}: {PARTY_NAMES[party_id]}" for party_id in PARTIPROGRAM_PATHS])
)

print(political_analyst_agent_prompt)
political_analyst_agent = Agent[None, list[PartyRecommendation]](
    model=model,
    instructions=political_analyst_agent_prompt,
    tools=party_expert_tools,
    output_type=list[PartyRecommendation],
)


result = political_analyst_agent.run_sync(
    "Jeg mener klimaet er vigtigt, hvilke partier prioriterer dette?",
    #"Det er vigtigt at bevare dansk kultur"
)

print(result.output)
print('========================================')
print(result.all_messages())

print(result.usage())