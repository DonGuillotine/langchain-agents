from langchain import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from decouple import config

OPENAI_API_KEY = config("OPENAI_API_KEY")

llm = OpenAI(
    openai_api_key=OPENAI_API_KEY,
    temperature=0
)

prompt = PromptTemplate(
  input_variables=["query"],
  template="You are helpful assistant. Help users with their important tasks, like a professor in a particular field. Query: {query}"
)


llm_chain = LLMChain(prompt=prompt, llm=llm)

print(llm_chain.run("Who is the current president of Nigeria?"))