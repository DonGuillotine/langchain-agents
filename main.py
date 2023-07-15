from langchain.agents import Tool
from langchain.agents import AgentType
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatAnthropic
from langchain.utilities import SerpAPIWrapper
from langchain.agents import initialize_agent
from decouple import config
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI

search = SerpAPIWrapper(serpapi_api_key=config("SERPAPI_API_KEY"))
tools = [
    Tool(
        name = "Current Search",
        func=search.run,
        description="useful for when you need to answer questions about current events or the current state of the world"
    ),
]

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

llm = ChatOpenAI(openai_api_key=config("OPENAI_API_KEY"), temperature=0)

agent_chain = initialize_agent(tools, llm, agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION, verbose=True, memory=memory)

agent_chain.run(input="what are some good dinners to make this week, if i like spicy food?")