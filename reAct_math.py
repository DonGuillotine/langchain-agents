from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.llms import OpenAI
from decouple import config
from dotenv import load_dotenv

load_dotenv()

llm = OpenAI(openai_api_key=config("OPENAI_API_KEY"), temperature=0)

tools = load_tools(["serpapi", "llm-math"], llm=llm)

agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

agent.run("Who is the President of Nigeria? What is her current age raised to the 2 power?")