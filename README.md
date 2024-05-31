# Generating-Cyber-Grooming-Chats-with-LLMs
Repository for the Individual Research Module at the University of Potsdam on cyber grooming detection in online Chats.
The goal of the project is to classify online chats into predatory and non-predatory using the multilingual model XLMRoBERTa. The training and evaluation of the model were conducted in 3 different settings to asses its ability to distinguish between cyber grooming and non-cyber grooming conversations in English, German, French and Dutch. The model was evaluated additionally using the CheckList framework to test some of its specific capabilities. The dataset used in the project is the publicly available PAN12 dataset, provided by the PAN Lab at the 2012 CLEF conference for a shared task on sexual predator identification.

## Setup 
Install all necessary packages with requirements.txt
```
$ pip install requirements.txt
```

Set your OpenAI API key:
```
# https://beta.openai.com/account/api-keys
export OPENAI_API_KEY=(YOUR OPENAI API KEY)
```

To generate chats using the few-shot prompting method, use the following commands:

For GPT-3.5-turbo:
```
python main.py --method few_shot_prompting --model gpt-3.5-turbo --num_chats 200 --examples_file Data/few_shot_examples.txt

```
For Wizard-Vicuna-uncensored:
```
python main.py --method few_shot_prompting --model Wizard-Vicuna-uncensored --num_chats 200 --examples_file Data/few_shot_examples.txt
```

To generate chats using the chain-of-thought prompting method, use the following commands:

For GPT-3.5-turbo:
```
python chain_of_thought_prompting.py --model gpt-3.5-turbo --num_chats 200
```
For Wizard-Vicuna-uncensored:
```
python chain_of_thought_prompting.py --model Wizard-Vicuna-uncensored --num_chats 200
```

To generate chats using the self-refine prompting method, use the following commands:

For GPT-3.5-turbo:
```
python self_refine_prompting.py --model gpt-3.5-turbo --num_chats 200
```
For Wizard-Vicuna-uncensored:
```
python self_refine_prompting.py --model Wizard-Vicuna-uncensored --num_chats 200
```
