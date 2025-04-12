from agno.agent import Agent
from agno.models.openai import OpenAIChat

agent = Agent(
    model=OpenAIChat(id="gpt-4o-mini"), # 12s - but results are better
    # model=OpenAIChat(id="gpt-3.5-turbo"), # 3s - but results are worse
    description="You are an enthusiastic news reporter with a flair for storytelling!",
    markdown=True
)

agent.print_response("Tell me about a breaking news story from Toronto.", stream=True)