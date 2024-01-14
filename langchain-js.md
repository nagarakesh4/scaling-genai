# Langchain.js

### Intro
- A RAG application needs LLM to perform a task, retrieve the text, tune the prompt, pass the LLM text output to structured for downstream steps of app
- Tools that connect the above workflow and also tune the workflow for e.g., changing the LLM models, tie multiple chains, these tools are called as Orchestrators.
- Langchain are popular ochestrator for LLM applications and helps to build them quickly

### Elements
- Data loaders - pull data from sources such as PDFs etc that augment the LLM generation
- Parsers - LLM operate with Natural language, while programming operate with formatted data, parsers extract and format Natural language output to create structured forms
- Prompts - Used to provide context for LLM
- Models - Provide abstraction layer on LLM (to write apps that are not vendor specific)
- Modules to support RAG - text splitters / integrations w/ vector stores
- LCEL - LangChain Expression lang - compose complex chain of these modules

### LCEL 
- composes chains of components
- components that implement this lang are called runnables
- allows input/output types, methods: invoke,stream, batch to use with LLM
- fallbacks/parallelism are available out-of-box


### Programming
```
import { ChatOpenAI } from "@langchain/openai";
import { HumanMessage } from "@langchain/core/messages";

const model = new ChatOpenAI({
    modelName: "gpt-3.5-turbo-1106"
});

await model.invoke([
    new HumanMessage("Tell me a fact")
]);
```

#### Building block: Prompt Template 
- calling models in isolation are useful as above
- convenient to factor logic into model inputs into reasonable parametrized components
- responsible for formatting user inputs that our model calls
```
import { ChatPromptTemplate } from "@langchain/core/prompts";

const prompt = ChatPromptTemplate.fromTemplate(
    `What are three good names for a company that makes {product}?`
)

await prompt.format({
    product: "colorful socks"
});  //string output - "Human: What are three good names for a company that makes colorful socks?" - this can be passed directly to LLM (not chat models)

//pass to Chatmodels
await prompt.formatMessages({
    product: "colorful socks"
}); //this will convert to HumanMessage{...content: ... } format for chat models, 

//to create both system and human messages for LLMs
import { 
  SystemMessagePromptTemplate, 
  HumanMessagePromptTemplate 
} from "@langchain/core/prompts";

const promptFromMessages = ChatPromptTemplate.fromMessages([
  SystemMessagePromptTemplate.fromTemplate(
    "You are an expert at picking company names."
  ),
  HumanMessagePromptTemplate.fromTemplate(
    "What are three good names for a company that makes {product}?"
  )
]);

await promptFromMessages.formatMessages({
    product: "shiny objects"
});

//using tuples with role in template: the above can be converted to, which also generates array of system and human messages
const promptFromMessages = ChatPromptTemplate.fromMessages([
  ["system", "You are an expert at picking company names."],
  ["human", "What are three good names for a company that makes {product}?"]
]);

await promptFromMessages.formatMessages({
  product: "shiny objects"
});

//the above can be passed to history, to inject history directly to prompt
```

#### LCEL
- elegant way of using prompts and models (instead of sending like above)
- composable syntax for chaining langchain models together
- objects that are compatible with LCEL are runnables
- can create simple chain

```
const chain = prompt.pipe(model);
await chain.invoke({
    product: "colorful socks"
}); //suggestes colorful socks names in chat messages
```

#### Output parser
```
import { StringOutputParser } from "@langchain/core/output_parsers";

const outputParser = new StringOutputParser();

//piping the above output parser (string) to model's prompt
const nameGenerationChain = prompt.pipe(model).pipe(outputParser);

await nameGenerationChain.invoke({
    product: "fancy cookies"
}); // prints only the string "1. Gourmet Cookie Co.\n2. DelicateDesserts\n3. Exquisite Eats Bakery"
```

- visual implementation of above [link](https://smith.langchain.com/public/9fd7689f-23c6-4ea1-8a76-fea1b5f8d5db/r/19a04da8-832f-4965-9587-72836a80dd4c)

#### Runnable sequence
- For more complex pipe chains, create an array of runnables

```
import { RunnableSequence } from "@langchain/core/runnables";

const nameGenerationChain = RunnableSequence.from([
    prompt,
    model,
    outputParser
]) ; //same as prev ex, but in a scalable array format

await nameGenerationChain.invoke({
    product: "fancy cookies"
}); //same response as above in string parser
```

#### Methods from LCEL 
##### Streaming (for Frontend)
- Another method from LCEL
- All sequences of runnables get a `.stream()` in an iterable stream
- rather than waiting for entire response, can show chunks as they get processed 
```
//using the above nameGenerationChain that has array of runnables implemented will help to get the string output instead of models response
//use .stream()
const stream = await nameGenerationChain.stream({
  product: "really cool robots",
}); //get indiv chunks
 
//loop over returned stream chunks
for await (const chunk of stream) {
    console.log(chunk);
}
```
#### Batch
- multiple/concurrent operations simultaneously
```
const inputs = [
    { product: "large calculators" },
    { product: "alpaca wool sweaters" }
];

//use .batch()
await nameGenerationChain.batch(inputs);

//batches output
[
  "1. MegaMath Calculators\n2. SummitTech Calculators\n3. CalculaGiant Company",
  "1. Alpaca Luxe\n2. Andean Elegance\n3. Woolly Alpaca Co."
]
```
