from langchain.llms import OpenAI
from langchain import PromptTemplate,LLMChain

import os



template = """Question: {question}

Answer: """
prompt = PromptTemplate(
        template=template,
    input_variables=['question']
)

# user question
question = "Which NFL team won the Super Bowl in the 2010 season?"
multi_template = """Answer the following questions one at a time.

Questions:
{questions}

Answers:
"""
long_prompt = PromptTemplate(template=multi_template, input_variables=["questions"])


qs_str = (
    "Which NFL team won the Super Bowl in the 2010 season?\n" +
    "If I am 6 ft 4 inches, how tall am I in centimeters?\n" +
    "Who was the 12th person on the moon?" +
    "How many eyes does a blade of grass have?"
)

davinci = OpenAI(model_name='text-davinci-003',openai_api_key='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')

llm_chain = LLMChain(
    prompt=prompt,
    llm=davinci
)

print(llm_chain.run(qs_str))