from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent
from langchain_openai import AzureChatOpenAI
from langchain_tavily import TavilySearch

tools = [
    TavilySearch()
]
llm = AzureChatOpenAI(
    temperature=0,
    model = "gpt-4o"
)

agent = create_agent(
    model=llm,
    tools=tools
)

def main():
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Serach for 3 job posting for an ai engineer the area of linkedin and list their results"    
                }
            ]
        }
    )

    print(result)


if __name__ == "__main__":
    main()