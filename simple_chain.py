from tempfile import template
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate, prompt
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt=PromptTemplate(
    template='Generate 5 main interesting facts about {topic}',
    input_variables=['topic']
)

model=ChatOpenAI()

parser=StrOutputParser()

chain = prompt | model | parser

result=chain.invoke({'topic':'cricket'})

# print(result)

# to visualize chain 
# chain.get_graph().print_ascii()

