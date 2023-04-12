from typing import Dict, Any, List
import openai
from prompts import AZURE_STUFF_PROMPT

class AzureChatGPT:
    def __init__(self, engine_name="gpt-35-turbo-0301-pretrained", **params):
        self.engine_name = engine_name
        self.temperature=params.get('temperature', 0)
        self.max_tokens=params.get('max_tokens', 800)
        self.top_p=params.get('top_p', 0.95)
        self.frequency_penalty=params.get('frequency_penalty',0)
        self.presence_penalty=params.get('presence_penalty', 0)
        self.stop=params.get('stop', ["<|im_end|>"])
        
        
#     def create_qa_prompt(self, question, summaries):

#         azure_qa_prompt = f"""
#         <|im_start|>system
#         Given the following extracted parts of a long document and a question, create a final answer with references ("SOURCES"). 
#         If you don't know the answer, just say that you don't know. Don't try to make up an answer.
#         ALWAYS return a "SOURCES" part in your answer. If you don't know the answer, leave "SOURCES" empty.
#         DO NOT modify the SOURCE information from the extracts.

#         QUESTION: What is the purpose of ARPA-H?
#         =========
#         CONTENT: More support for patients and families. \n\nTo get there, I call on Congress to fund ARPA-H, the Advanced Research Projects Agency for Health. \n\nIt’s based on DARPA—the Defense Department project that led to the Internet, GPS, and so much more.  \n\nARPA-H will have a singular purpose—to drive breakthroughs in cancer, Alzheimer’s, diabetes, and more.
#         SOURCE: C00020
#         CONTENT: While we’re at it, let’s make sure every American can get the health care they need. \n\nWe’ve already made historic investments in health care. \n\nWe’ve made it easier for Americans to get the care they need, when they need it. \n\nWe’ve made it easier for Americans to get the treatments they need, when they need them. \n\nWe’ve made it easier for Americans to get the medications they need, when they need them.
#         SOURCE: C00022
#         CONTENT: The V.A. is pioneering new ways of linking toxic exposures to disease, already helping  veterans get the care they deserve. \n\nWe need to extend that same care to all Americans. \n\nThat’s why I’m calling on Congress to pass legislation that would establish a national registry of toxic exposures, and provide health care and financial assistance to those affected.
#         SOURCE: C00023
#         =========
#         FINAL ANSWER: The purpose of ARPA-H is to drive breakthroughs in cancer, Alzheimer’s, diabetes, and more.
#         SOURCES: C00020

#         QUESTION: What did the president say about Michael Jackson?
#         =========
#         CONTENT: Madam Speaker, Madam Vice President, our First Lady and Second Gentleman. Members of Congress and the Cabinet. Justices of the Supreme Court. My fellow Americans.  \n\nLast year COVID-19 kept us apart. This year we are finally together again. \n\nTonight, we meet as Democrats Republicans and Independents. But most importantly as Americans. \n\nWith a duty to one another to the American people to the Constitution. \n\nAnd with an unwavering resolve that freedom will always triumph over tyranny.
#         SOURCE: F00010
#         FINAL ANSWER: The president did not mention Michael Jackson.
#         SOURCES:
#         <|im_end|>
#         <|im_start|>user
#         QUESTION: {question}
#         =========
#         {summaries}
#         =========
#         FINAL ANSWER:"""

#         return azure_qa_prompt


    def complete(self, question, top_docs):
        summaries = '\n\n'.join(['Content: ' + d.page_content + '\n' + 'Source:' + d.metadata['source'] for d in top_docs])
        
        # qa_prompt = self.create_qa_prompt(question, summaries)
        qa_prompt=  AZURE_STUFF_PROMPT.format(summaries=summaries, question=question)
        response = openai.Completion.create(
            engine=self.engine_name,
            prompt=qa_prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
            stop=self.stop
        )
        
        
        print(qa_prompt)
        print('\n\n')
        print(response)
        
#         print(self.temperature)
#         print(self.top_p)
#         print(self.frequency_penalty)
#         print(self.presence_penalty)

        return {"output_text": response['choices'][0]['text'].strip()}


