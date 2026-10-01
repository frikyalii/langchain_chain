# in this sequential chain we taking
#  prompt input >prompt template > chatopenAi > strOutputParser > StrOutputParserOutput > PromptTemplate |> chatOpenAI > StrOutputParser > StrOutputParserOutput
 


from tempfile import template
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate, prompt
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables  import RunnableParallel
load_dotenv()

prompt1=PromptTemplate(
    template='Generate short and simple notes from following text \n {text}',
    input_variables=['topic']
)


prompt2=PromptTemplate(
    template='Generate 5 short question answer from following text {text}',
    input_variables=['text']
)

prompt3=PromptTemplate(
    template='Merge provided notes and quiz into single documents \n notes-> {notes} and quiz-> {quiz}',
    input_variables=['notes','quiz']
)

model1=ChatOpenAI()

model2=ChatOpenAI()

parser=StrOutputParser()


parallel_chain=RunnableParallel({
    'notes': prompt1 | model1 | parser,
    'quiz': prompt2 | model2 | parser
})

merge_chain=prompt3 | model1 | parser
 
chain=parallel_chain | merge_chain

text="""
Bollywood is the Hindi-language film industry based in Mumbai (formerly Bombay). The name blends "Bombay" and "Hollywood" and became popular in the 1970s.
It is part of Indian cinema, alongside Tollywood (Telugu), Kollywood (Tamil), Mollywood (Malayalam), Sandalwood (Kannada), and others.
Major hub: Film City (Dadasaheb Phalke Chitranagari), Goregaon, Mumbai.
India makes more films per year than any other country (roughly 1,500 to 2,000).
Historical Timeline
1913: Raja Harishchandra by Dadasaheb Phalke, the first full-length Indian feature film (silent). He is called the "Father of Indian Cinema".
1931: Alam Ara by Ardeshir Irani, the first Indian talkie.
1937: Kisan Kanya, the first Indian colour film.
1940s to 60s (Golden Age): Raj Kapoor, Dilip Kumar, Dev Anand, Guru Dutt, Nargis, Madhubala, Meena Kumari. Key films: Awaara (1951), Pyaasa (1957), Mother India (1957), Mughal-e-Azam (1960).
1970s: Rise of the "Angry Young Man", Amitabh Bachchan. Zanjeer (1973), Deewaar (1975), and Sholay (1975, directed by Ramesh Sippy). Writers Salim-Javed. Rajesh Khanna was called the first superstar (Aradhana, 1969).
1980s: Masala and action films, plus the parallel cinema movement (Shyam Benegal, Govind Nihalani). Mr. India (1987).
1990s: Economic liberalisation, family romances, NRI themes. Hum Aapke Hain Koun..! (1994), Dilwale Dulhania Le Jayenge (1995, Aditya Chopra), Kuch Kuch Hota Hai (1998). The three Khans (Shah Rukh, Aamir, Salman) dominated. Film industry status was granted in 1998.
2000s: Multiplexes, corporate funding, and overseas markets grew. Lagaan (2001), Devdas (2002), Munna Bhai M.B.B.S. (2003), 3 Idiots (2009).
2010s: The ₹100 crore and ₹200 crore clubs. Dangal (2016), PK, Bajrangi Bhaijaan. Parallel rise of pan-Indian films such as Baahubali.
2020s: COVID disruption, the OTT boom, and pan-Indian blockbusters (RRR, KGF, Pushpa). Hindi hits include Pathaan and Jawan (2023) and Stree 2 (2024).
"""

result=chain.invoke({'text':text })
print(result)

chain.get_graph().print_ascii()


#   +---------------------------+            
#             | Parallel<notes,quiz>Input |            
#             +---------------------------+            
#                  **               **                 
#               ***                   ***              
#             **                         **            
# +----------------+                +----------------+ 
# | PromptTemplate |                | PromptTemplate | 
# +----------------+                +----------------+ 
#           *                               *          
#           *                               *          
#           *                               *          
#   +------------+                    +------------+   
#   | ChatOpenAI |                    | ChatOpenAI |   
#   +------------+                    +------------+   
#           *                               *          
#           *                               *          
#           *                               *          
# +-----------------+              +-----------------+ 
# | StrOutputParser |              | StrOutputParser | 
# +-----------------+              +-----------------+ 
#                  **               **                 
#                    ***         ***                   
#                       **     **                      
#            +----------------------------+            
#            | Parallel<notes,quiz>Output |            
#            +----------------------------+            
#                           *                          
#                           *                          
#                           *                          
#                  +----------------+                  
#                  | PromptTemplate |                  
#                  +----------------+                  
#                           *                          
#                           *                          
#                           *                          
#                    +------------+                    
#                    | ChatOpenAI |                    
#                    +------------+                    
#                           *                          
#                           *                          
#                           *                          
#                 +-----------------+                  
#                 | StrOutputParser |                  
#                 +-----------------+                  
#                           *                          
#                           *                          
#                           *                          
#               +-----------------------+              
#               | StrOutputParserOutput |              
#               +-----------------------+    