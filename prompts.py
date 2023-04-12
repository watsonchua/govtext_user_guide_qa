from langchain.prompts import PromptTemplate

## Use a shorter template to reduce the number of tokens in the prompt
template = """Create a final answer to the given questions using the provided document excerpts(in no particular order) as references. ALWAYS include a "SOURCES" section in your answer including only the minimal set of sources needed to answer the question. If you are unable to answer the question, simply state that you do not know. Do not attempt to fabricate an answer and leave the SOURCES section empty.

---------

QUESTION: What is the purpose of ARPA-H?
=========
Content: More support for patients and families. \n\nTo get there, I call on Congress to fund ARPA-H, the Advanced Research Projects Agency for Health. \n\nIt’s based on DARPA—the Defense Department project that led to the Internet, GPS, and so much more.  \n\nARPA-H will have a singular purpose—to drive breakthroughs in cancer, Alzheimer’s, diabetes, and more.
Source: 1-32
Content: While we’re at it, let’s make sure every American can get the health care they need. \n\nWe’ve already made historic investments in health care. \n\nWe’ve made it easier for Americans to get the care they need, when they need it. \n\nWe’ve made it easier for Americans to get the treatments they need, when they need them. \n\nWe’ve made it easier for Americans to get the medications they need, when they need them.
Source: 1-33
Content: The V.A. is pioneering new ways of linking toxic exposures to disease, already helping  veterans get the care they deserve. \n\nWe need to extend that same care to all Americans. \n\nThat’s why I’m calling on Congress to pass legislation that would establish a national registry of toxic exposures, and provide health care and financial assistance to those affected.
Source: 1-30
=========
FINAL ANSWER: The purpose of ARPA-H is to drive breakthroughs in cancer, Alzheimer’s, diabetes, and more.
SOURCES: 1-32

---------

QUESTION: Where does Watson live?
=========
Content: Watson was born in Singapore in 1983.
Source: C00001
Content: Watson is living together with his parents in a flat in Jurong West.
Source: C00002
Content: Watson works in GovTech as a data scientist.
Source: C00003
=========
FINAL ANSWER: "Watson lives in Jurong West, Singapore." 
SOURCES: C00001, C00002

---------

QUESTION: {question}
=========
{summaries}
=========
FINAL ANSWER:"""

STUFF_PROMPT = PromptTemplate(
    template=template, input_variables=["summaries", "question"]
)

# ---------

# QUESTION: What did the president say about Michael Jackson?
# =========
# Content: Madam Speaker, Madam Vice President, our First Lady and Second Gentleman. Members of Congress and the Cabinet. Justices of the Supreme Court. My fellow Americans.  \n\nLast year COVID-19 kept us apart. This year we are finally together again. \n\nTonight, we meet as Democrats Republicans and Independents. But most importantly as Americans. \n\nWith a duty to one another to the American people to the Constitution. \n\nAnd with an unwavering resolve that freedom will always triumph over tyranny.
# Source: F00010
# =========
# FINAL ANSWER: "I don't know." 
# SOURCES:

azure_template = """<|im_start|>system
You are an question answering assistant. Your job is to answer questions given a context and give the sources of information.
Given the following extracted parts of a long document and a question, create a final answer with references ("SOURCES").
If you don't know the answer, say "I don't know. Please direct the question to ITSM." Don't try to make up an answer.
ALWAYS return a "SOURCES" part in your answer. If you don't know the answer, leave "SOURCES" empty.
DO NOT modify the SOURCE information from the extracts.
<|im_end|>
<|im_start|>user
QUESTION: What is the purpose of ARPA-H?
=========
Content: More support for patients and families. \n\nTo get there, I call on Congress to fund ARPA-H, the Advanced Research Projects Agency for Health. \n\nIt’s based on DARPA—the Defense Department project that led to the Internet, GPS, and so much more.  \n\nARPA-H will have a singular purpose—to drive breakthroughs in cancer, Alzheimer’s, diabetes, and more.
Source: C00020
Content: While we’re at it, let’s make sure every American can get the health care they need. \n\nWe’ve already made historic investments in health care. \n\nWe’ve made it easier for Americans to get the care they need, when they need it. \n\nWe’ve made it easier for Americans to get the treatments they need, when they need them. \n\nWe’ve made it easier for Americans to get the medications they need, when they need them.
Source: C00022
Content: The V.A. is pioneering new ways of linking toxic exposures to disease, already helping  veterans get the care they deserve. \n\nWe need to extend that same care to all Americans. \n\nThat’s why I’m calling on Congress to pass legislation that would establish a national registry of toxic exposures, and provide health care and financial assistance to those affected.
Source: C00023
=========
<|im_end|>
<|im_start|>assistant
FINAL ANSWER: The purpose of ARPA-H is to drive breakthroughs in cancer, Alzheimer’s, diabetes, and more.
SOURCES: C00020
<|im_end|>
<|im_start|>user
QUESTION: Where does Watson live?
=========
Content: Watson was born in Singapore in 1983.
Source: C00001
Content: Watson is living together with his parents in a flat in Jurong West.
Source: C00002
Content: Watson works in GovTech as a data scientist.
Source: C00003
=========
<|im_end|>
<|im_start|>assistant
FINAL ANSWER: "Watson lives in Jurong West, Singapore." 
SOURCES: C00001, C00002
<|im_end|>
<|im_start|>user
QUESTION: What did the president say about Michael Jackson?
=========
Content: Madam Speaker, Madam Vice President, our First Lady and Second Gentleman. Members of Congress and the Cabinet. Justices of the Supreme Court. My fellow Americans.  \n\nLast year COVID-19 kept us apart. This year we are finally together again. \n\nTonight, we meet as Democrats Republicans and Independents. But most importantly as Americans. \n\nWith a duty to one another to the American people to the Constitution. \n\nAnd with an unwavering resolve that freedom will always triumph over tyranny.
Source: F00010
<|im_end|>
<|im_start|>assistant
FINAL ANSWER: "I don't know." 
SOURCES:
<|im_end|>
<|im_start|>user
QUESTION: {question}
=========
{summaries}
=========
<|im_end|>
<|im_start|>assistant
FINAL ANSWER:
"""

# FINAL ANSWER:"""

# QUESTION: What did the president say about Michael Jackson?
# =========
# CONTENT: Madam Speaker, Madam Vice President, our First Lady and Second Gentleman. Members of Congress and the Cabinet. Justices of the Supreme Court. My fellow Americans.  \n\nLast year COVID-19 kept us apart. This year we are finally together again. \n\nTonight, we meet as Democrats Republicans and Independents. But most importantly as Americans. \n\nWith a duty to one another to the American people to the Constitution. \n\nAnd with an unwavering resolve that freedom will always triumph over tyranny.
# SOURCE: F00010
# FINAL ANSWER: The president did not mention Michael Jackson.
# SOURCES:

AZURE_STUFF_PROMPT = PromptTemplate(
    template=azure_template, input_variables=["summaries", "question"]
)