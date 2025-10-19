import os
from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent
from langchain_openai import AzureChatOpenAI
from langchain_tavily import TavilySearch
from langchain.agents.structured_output import ToolStrategy
from langchain.tools import tool

from schemas import AgentResponse

@tool
def get_text_length(text: str) -> int:
    """Returns the length of a text by characters"""
    text = text.strip("'\n").strip(
        '"'
    )  # stripping away non alphabetic characters just in case
    return len(text)

tools = [TavilySearch(), get_text_length]
llm = AzureChatOpenAI(temperature=0, model="gpt-4o")


agent = create_agent(model=llm, tools=tools, response_format=ToolStrategy(AgentResponse))

def main():
    result = agent.invoke({"messages": [{"role": "user", "content": "What is the length of the word: Langchain"}]})
    # print(result['messages'])
    print(result['structured_response'])

if __name__ == "__main__":
    main()
