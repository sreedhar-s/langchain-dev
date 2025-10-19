from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent
from langchain_openai import AzureChatOpenAI
from langchain_tavily import TavilySearch
from langchain.agents.structured_output import ToolStrategy

from schemas import AgentResponse

tools = [TavilySearch()]
llm = AzureChatOpenAI(temperature=0, model="gpt-4o")

agent = create_agent(model=llm, tools=tools, response_format=ToolStrategy(AgentResponse))

def main():
    result = agent.invoke({"messages": [{"role": "user", "content": "search for 3 job postings for an ai engineer using langchain in the bay area on linkedin and list their details"}]})
    print(result)

if __name__ == "__main__":
    main()
