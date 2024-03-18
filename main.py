import os
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI

load_dotenv(find_dotenv(), override=True)

os.environ.get('OPENAI_API_KEY')


llm = ChatOpenAI()
# output = llm.invoke("Who is ALbert Eintein?", model='gpt-3.5-turbo')
# help(ChatOpenAI)
# print(output.content)

from langchain.schema import(SystemMessage, AIMessage, HumanMessage)

messages = [
  SystemMessage(content='You are a physicist and respond only in German'),
  HumanMessage(content='Explain quantum mechanics in one sentence')
]

# output = llm.invoke(messages)
# print(output)

######### in Memort Caching
from langchain.globals import set_llm_cache
from langchain_openai import OpenAI
llm =  OpenAI(model_name='gpt-3.5-turbo-instruct')

from langchain.cache import InMemoryCache
set_llm_cache(InMemoryCache())
prompt= 'Tell ne a joke that a toddler can understand'

# llm.invoke(prompt)

##### SQLite Caching

from langchain.cache import SQLiteCache
# set_llm_cache(SQLiteCache(database_path='.langchain.db'))
#First request takes longer, not in chace
# llm.invoke("Tell me a joke")

# Second request (cached, faster)
# llm.invoke("Tell me a joke")


##### LLM Streaming

from langchain_openai import ChatOpenAI

llm = ChatOpenAI()
prompt = 'Write a rock song about the Moon and a Raven'
# print(llm.invoke(prompt).content)
# llm.stream(prompt)
# for chunk in llm.stream(prompt):
  # print(chunk.content, end='', flush=True)
  

###### Prompt Templates

from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

template = '''You are an experienced virologist.
Write a few sentences about the following virus "{virus} in {language}" '''

# prompt_template = PromptTemplate.from_template(template=template)

# prompt = prompt_template.format(virus='hiv', language='german')
# # prompt

# llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0)
# output = llm.invoke(prompt)

# print(ouput.content)


##### ChatPromptTemplates

from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage

chat_template = ChatPromptTemplate.from_messages(
  [
    SystemMessage(content='You respond ony in Json format'),
    HumanMessagePromptTemplate.from_template('Top {n} countries in {area} by pouplation')
  ]
)

messages = chat_template.format_messages(n='10', area='Europe')

# print(messages)

from langchain_openai import ChatOpenAI
# llm = ChatOpenAI()
# ouput = llm.invoke(messages)
# print(output.content)

###### Simple Chains
from langchain_openai import ChatOpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain

llm = ChatOpenAI()
template = '''You are an experience virologist.
Write a few sentences about the following virus "{virus}" in {language}.'''
prompt_template = PromptTemplate.from_template(template=template)

chain = LLMChain(
    llm=llm,
    prompt=prompt_template,
    verbose=True
)

output = chain.invoke({'virus': 'HSV', 'language': 'Spanish'})

template = 'What is the capital of {country}?. List the top 3 places to visit in that city. Use bullet points'
prompt_template = PromptTemplate.from_template(template=template)

# Initialize an LLMChain with the ChatOpenAI model and the prompt template
chain = LLMChain(
    llm=llm,
    prompt=prompt_template,
    verbose=True
)

country = input('Enter Country: ')

# Invoke the chain with specific virus and language values
# output = chain.invoke(country)
# print(output['text'])


#####    Sequential Chains
from langchain_openai import ChatOpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain

# Initialize the first ChatOpenAI model (gpt-3.5-turbo) with specific temperature
llm1 = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.5)

# Define the first prompt template
prompt_template1 = PromptTemplate.from_template(
    template='You are an experienced scientist and Python programmer. Write a function that implements the concept of {concept}.'
)
# Create an LLMChain using the first model and the prompt template
chain1 = LLMChain(llm=llm1, prompt=prompt_template1)

# Initialize the second ChatOpenAI model (gpt-4-turbo) with specific temperature
llm2 = ChatOpenAI(model_name='gpt-4-turbo-preview', temperature=1.2)

# Define the second prompt template
prompt_template2 = PromptTemplate.from_template(
    template='Given the Python function {function}, describe it as detailed as possible.'
)
# Create another LLMChain using the second model and the prompt template
chain2 = LLMChain(llm=llm2, prompt=prompt_template2)

# Combine both chains into a SimpleSequentialChain
overall_chain = SimpleSequentialChain(chains=[chain1, chain2], verbose=True)

# Invoke the overall chain with the concept "linear regression"
output = overall_chain.invoke('linear regression')


from langchain_experimental.utilities import PythonREPL
python_repl = PythonREPL()
python_repl.run('print([n for n in range(1, 100) if n % 13 == 0])')


from langchain_experimental.agents.agent_toolkits import create_python_agent
from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain_openai import ChatOpenAI

# Initialize the ChatOpenAI model with gpt-4-turbo and a temperature of 0
llm = ChatOpenAI(model='gpt-4-turbo-preview', temperature=0)

# Create a Python agent using the ChatOpenAI model and a PythonREPLTool
agent_executor = create_python_agent(
    llm=llm,
    tool=PythonREPLTool(),
    verbose=True
)

# Invoke the agent
prompt = 'Calculate the square root of the factorial of 12 and display it with 4 decimal points'
agent_executor.invoke(prompt)

response = agent_executor.invoke('What is the answer to 5.1 ** 7.3?')