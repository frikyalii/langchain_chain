# in this sequential chain we taking
#  prompt input >prompt template > chatopenAi > strOutputParser > StrOutputParserOutput > PromptTemplate |> chatOpenAI > StrOutputParser > StrOutputParserOutput
 


from tempfile import template
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate, prompt
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt1=PromptTemplate(
    template='Generate detail report about {topic}',
    input_variables=['topic']
)


prompt2=PromptTemplate(
    template='Generate 5 main interesting summary  about {text}',
    input_variables=['text']
)

model=ChatOpenAI()

parser=StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser

result=chain.invoke({'topic':'unemployement'})

print(result)

chain.get_graph().print_ascii()

