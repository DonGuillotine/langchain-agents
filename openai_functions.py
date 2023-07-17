from langchain import LLMMathChain, OpenAI, SerpAPIWrapper
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from decouple import config


llm = ChatOpenAI(openai_api_key=config("OPENAI_API_KEY"), temperature=0, model="gpt-3.5-turbo-0613")
search = SerpAPIWrapper(serpapi_api_key=config("SERPAPI_API_KEY"))
llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)
tools = [
    Tool(
        name = "Search",
        func=search.run,
        description="useful for when you need to answer questions about current events. You should ask targeted questions"
    ),
    Tool(
        name="Calculator",
        func=llm_math_chain.run,
        description="useful for when you need to answer questions about math"
    )
]

agent = initialize_agent(tools, llm, agent=AgentType.OPENAI_FUNCTIONS, verbose=True)

agent.run("Who is the President of Nigeria? What is his current age raised to the 2 power?")