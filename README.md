## Models, Prompts, Parsers, Memory, Chains, Conversation/Prediction
[ref](https://learn.deeplearning.ai/langchain)

### direct API calls to OpenAI.
```
import os
import openai

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']

llm_model = "gpt-3.5-turbo"

def generateOpenAiResponse(prompt, model=llm_model):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0, 
    )
    return response # should be parsed at response.choices[0].content
  
customer_review = "The headphones are good, however \
I would have preferred more options to configure for the price I paid, I will give it a 3 star";
desired_tone = "American english in a calm and respectful tone";
prompt = """Translate the review {customer_review} into ${desired_tone}"""
generateOpenAiResponse(prompt, llm_model)
```

```
response:
<OpenAIObject chat.completion id=chatcmpl-8b6TFGpehcU0LBz7np2vTddT2S8Gk at 0x7f90758249f0> JSON: {
  "choices": [
    {
      "finish_reason": "stop",
      "index": 0,
      "logprobs": null,
      "message": {
        "content": "While the headphones may not be the best, I do feel that for the price I paid, there could have been fewer options to configure. Despite this, I still give them a 10-star rating.",
        "role": "assistant"
      }
    }
  ],
  "created": 1703853701,
  "id": "chatcmpl-8b6TFGpehcU0LBz7np2vTddT2S8Gk",
  "model": "gpt-3.5-turbo-0301",
  "object": "chat.completion",
  "system_fingerprint": null,
  "usage": {
    "completion_tokens": 41,
    "prompt_tokens": 51,
    "total_tokens": 92
  }
}

```

```
# Models: Using lang chain
from langchain.chat_models import ChatOpenAI

# To control the randomness and creativity of the generated
# text by an LLM, use temperature = 0.0
chat = ChatOpenAI(temperature=0.0, model=llm_model)

prompt_template = """Translate the review {customer_email} into {style}"""

# create langChain prompt template from the given prompt_template
from langchain.prompts import ChatPromptTemplate
langChainPrompt_template = ChatPromptTemplate.from_template(prompt_template)

langChainPrompt_template.messages[0].prompt.input_variables # customer_email, style are the arguments to be passed to langchain

# pass in the dynamic parameters specified in the prompt to properly format the prompt message
customer_messages = langChainPrompt_template.format_messages(
                    style=desired_tone,
                    customer_email=customer_review)

print(customer_messages[0]) # formats the prompt with the given dynamic variables as seen in open ai prompt line 23

# Call the LLM to translate to the style of the customer message
customer_response = chat(customer_messages)

print(customer_response.content)

```

### Parsers: converting review to JSON format
```
customer_review = """\
This leaf blower is pretty amazing.  It has four settings:\
candle blower, gentle breeze, windy city, and tornado. \
It arrived in two days, just in time for my wife's \
anniversary present. \
I think my wife liked it so much she was speechless. \
So far I've been the only one using it, and I've been \
using it every other morning to clear the leaves on our lawn. \
It's slightly more expensive than the other leaf blowers \
out there, but I think it's worth it for the extra features.
"""
```

```
prompt_template = """\
For the following customer_review, extract the following information:

gift: Was the item purchased as a gift for someone else? \
Answer True if yes, False if not or unknown.

delivery_days: How many days did it take for the product \
to arrive? If this information is not found, output -1.

price_value: Extract any sentences about the value or price,\
and output them as a comma separated Python list.

Format the output as JSON with the following keys:
gift
delivery_days
price_value

customer_review: {customer_review}
"""
```
```
# almost same as previous usecase, except that we explicitly state the desired json format in the template and probably
# won't need that second dynamic variable (style) while formatting the prompt

# Using lang chain
from langchain.chat_models import ChatOpenAI

# To control the randomness and creativity of the generated
# text by an LLM, use temperature = 0.0
chat = ChatOpenAI(temperature=0.0, model=llm_model)

#create langchain prompt template from the given prompt_template
from langchain.prompts import ChatPromptTemplate
langChainPrompt_template = ChatPromptTemplate.from_template(prompt_template)

# pass in the dynamic parameters specified in the prompt to properly format the prompt message
customer_messages = langChainPrompt_template.format_messages(
                    customer_review=customer_review)

# Call the LLM to translate to the style of the customer message
customer_response = chat(customer_messages)

print(customer_response.content) # prints in json format however
type(response.content) # is still a string, so cannot really do a response.content.get('gift')


# Parse the LLM output string into a Python dictionary

from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser

# declare a schema on how the JSON should be
gift_schema = ResponseSchema(name="gift",
                             description="Was the item purchased\
                             as a gift for someone else? \
                             Answer True if yes,\
                             False if not or unknown.")
delivery_days_schema = ResponseSchema(name="delivery_days",
                                      description="How many days\
                                      did it take for the product\
                                      to arrive? If this \
                                      information is not found,\
                                      output -1.")

response_json = [gift_schema, delivery_days_schema]

output_parsed_json = StructuredOutputParser.from_response_schemas(response_json)

formatted_json = output_parsed_json.get_format_instructions()

# Include the above. formatted json in the prompt - without this it cannot really do a JSON format
json_formatted_prompt_template = """\
For the following customer_review, extract the following information:

customer_review:{customer_review}

{formatted_json}
"""

# the above template will have json instructions as part of {formatted_json}

# Call the LLM to translate to the style of the customer message
response = chat(messages)

# use the output parsed json to extract the keys after parsing
output_json = output_parsed_json.parse(response.content)

type(output_json) #dict

output_json.get('delivery_days') # 2
```

### LangChain: Memory and Conversation/Prediction

- ConversationBufferMemory
- ConversationBufferWindowMemory
- ConversationTokenBufferMemory
- ConversationSummaryMemory

```
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

llm = ChatOpenAI(temperature=0.0, model=llm_model)

memory = ConversationBufferMemory()
conversation = ConversationChain(
    llm=llm, 
    memory = memory,
    verbose=True
)

conversation.predict(input="Hi, my name is Venkata")
conversation.predict(input="Hi, 1 + 1 is 2 )
conversation.predict(input="Hi, what is my name?")

# memory buffer of so far stored conv                 
print(memory.buffer)
                     
# manually store data to memory
memory = ConversationBufferMemory()
memory.save_context({"input": "Hi"}, {"output": "What's up"})

# load entire memory in json
memory.load_memory_variables({})

# -------- stores into memory based on the buffer window size
memory = ConversationBufferWindowMemory(k=1)
memory.save_context({"input": "Hi"}, {"output": "What's up"})
memory.save_context({"input": "Not much, just hanging"}, {"output": "Cool"})

print(memory.buffer) # prints only the latest saved memory context (as window size is 1)

                     
# -------- stores into memory based on the available memory tokens
from langchain.memory import ConversationTokenBufferMemory
memory = ConversationTokenBufferMemory(llm=llm, max_token_limit=50) 

# -------- stores into memory the summarized version
from langchain.memory import ConversationSummaryBufferMemory

memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=100)
memory.save_context({"input": "Hello"}, {"output": "What's up"})
memory.save_context({"input": "Not much, just hanging"}, {"output": "Cool"})
                     

# create a long string and store it in memory
schedule = "There is a meeting at 8am with your product team. \
You will need your powerpoint presentation prepared. \
9am-12pm have time to work on your LangChain \
project which will go quickly because Langchain is such a powerful tool. \
At Noon, lunch at the italian resturant with a customer who is driving \
from over an hour away to meet you to understand the latest in AI. \
Be sure to bring your laptop to show the latest LLM demo."
memory.save_context({"input": "What is on the schedule today?"}, {"output": f"{schedule}"})

# Prints a summarized version of the above details
memory.load_memory_variables({})

# lang chain coversation
conversation = ConversationChain(
    llm=llm, 
    memory = memory,
    verbose=False
)
                     
conversation.predict(input="What would be a good demo to show?")

# also memorizes the prediction the AI has provided
memory.load_memory_variables({})
```

### LLM Chains: 
- important building block of langchain
- combines llm with prompt and puts all the building blocks together to carry out a sequence of operations on the input text.

```
# pipe install pandas to read csv file
import pandas as pd
df = pd.read_csv('Data.csv')

# print top 5 rows
df.head()

# Code
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

llm = ChatOpenAI(temperature=0.9, model=llm_model)

langChainPrompt_template = ChatPromptTemplate.from_template(
    "What is the best name to describe \
    a company that makes {product}?"
)

# combine the above llm and prompt into a chain using LLMChain
chain = LLMChain(llm=llm, prompt=langChainPrompt_template)

product = "Queen Size Sheet Set for babies"
chain.run(product) # suggests as Royal Linens
```

### Simple Sequential Chain
- run one after the other chain
- output of one chain is the input to the next chain
- simple sequential - single input / output
- sequential - multiple inputs/outputs

```
# Single Sequential chain

from langchain.chains import SimpleSequentialChain

llm = ChatOpenAI(temperature=0.9, model=llm_model)

# prompt template 1
first_prompt = ChatPromptTemplate.from_template(
    "What is the best name to describe \
    a company that makes {product}?"
)

# Chain 1
chain_one = LLMChain(llm=llm, prompt=first_prompt)

# prompt template 2
second_prompt = ChatPromptTemplate.from_template(
    "Write a 20 words description for the following \
    company:{company_name}"
)
# chain 2
chain_two = LLMChain(llm=llm, prompt=second_prompt)

# combining chain 1 and 2
overall_simple_chain = SimpleSequentialChain(chains=[chain_one, chain_two],verbose=True)

# run the model
overall_simple_chain.run(product) # here run is same as other ways to call the model to run,
however it also needs the positional arguments to be provided in this case `product` (dynamic)
```

```
# Regular Sequential Chain (multiple chains, reusing from previous)
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import SequentialChain
from langchain.chains import LLMChain

# temperature 0.9 will let the ai to generate message with more text in the content
llm = ChatOpenAI(temperature=0.0, model=llm_model)

# prompt template 1: translate to english
first_prompt = ChatPromptTemplate.from_template(
    "Translate the following review to english:"
    "\n\n{Review}"
)

# chain 1: input= Review and output= English_Review
chain_one = LLMChain(llm=llm, prompt=first_prompt, output_key="English_Review")

# prompt template 2: generate a summary of the review
second_prompt = ChatPromptTemplate.from_template(
    "Can you summarize the following review in 1 sentence:"
    "\n\n{English_Review}"
)

# chain 2: input= English_Review and output= summary
chain_two = LLMChain(llm=llm, prompt=second_prompt, output_key="summary")

# prompt template 3: detect what language is the review in
third_prompt = ChatPromptTemplate.from_template(
    "What language is the following review:\n\n{Review}"
)

# chain 3: input= Review and output= language
chain_three = LLMChain(llm=llm, prompt=third_prompt, output_key="language")

# # prompt template 4: generate a follow up message in the original language
fourth_prompt = ChatPromptTemplate.from_template(
    "Write a follow up response to the following "
    "summary in the specified language:"
    "\n\nSummary: {summary}\n\nLanguage: {language}"
)

# chain 4: input= summary, language and output= followup_message
chain_four = LLMChain(llm=llm, prompt=fourth_prompt, output_key="followup_message")

# overall_chain: input= Review 
# and output= English_Review,summary, followup_message
overall_chain = SequentialChain(
    chains=[chain_one, chain_two, chain_three, chain_four],
    input_variables=["Review"],
    output_variables=["English_Review", "summary","followup_message"],
    verbose=True
)

# choose a review from dataset or provide your own
#review = df.Review[5]
# review = 'హెడ్‌ఫోన్‌లు అస్సలు బాగోలేదు, తక్కువ ధరకు వస్తే వాటిని కొంటాను'
review = 'हेडफोन अच्छे नहीं हैं, लेकिन अगर मुझे ये कम कीमत पर मिलेंगे तो मैं इन्हें खरीदूंगा'


# pass it into the chain
overall_chain(review)
```
```
{'Review': 'हेडफोन अच्छे नहीं हैं, लेकिन अगर मुझे ये कम कीमत पर मिलेंगे तो मैं इन्हें खरीदूंगा',
 'English_Review': 'The headphones are not good, but if I get them at a lower price, I will buy them.',
 'summary': 'The reviewer would only purchase the headphones if they were available at a discounted price.',
 'language': 'Hindi',
 'followup_message': 'सारांश: रिव्यूअर केवल उन हेडफोन को खरीदेंगे अगर उन्हें छूट की कीमत में उपलब्ध हों।\n\nउत्तर: यह अच्छी बात है कि आपको यह हेडफोन पसंद आया है। हम आपको बताना चाहेंगे कि इस समय इस हेडफोन के लिए छूट की पेशकश उपलब्ध है। आप इसे अपनी आवश्यकताओं के अनुसार खरीद सकते हैं। आप इसे ऑनलाइन या ऑफलाइन दोनों तरीकों से खरीद सकते हैं। हम आशा करते हैं कि आप इसे खरीदने के लिए तैयार होंगे।'}
```

### Routing Chain
- if there are multiple sub chains, we can have a router chain to decide where the request should go to, different prompt template, with different prompt template write more info about them (in array), this info will go to router chain so that router chain can decide when to use the sub-chain
- MultiPromptChain - specific type of chain - routing b/n multiple different prompt templates
- LLMRouterChain - uses language model to route b/n diff sub chains (uses the info array)
- RouterOutputParser - parses LLM output into dictionary that can be used in downstream to decide which stream to use and what input should be

### Agents
- LLM is a reasoning change, will use the background knowledge from internet or our custom response to decided what to do next
- Agents are new to the field
- incorporated to search engine (comes with lang chain), integrate with data store
- 
