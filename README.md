# Generating-Cyber-Grooming-Chats-with-LLMs
The goal of this project is to leverage large language models (LLMs) to generate cyber grooming chats. We examine three types of prompting methods - few-shot promoting, chain-of-thought promoting, and iterative-self-refine promoting - on two LLMs: GPT-3.5-turbo and Wizard-Vicuna-30B-uncensored. We assess the quality of the generated chats using a 5-point rating system on three attributes, combining both human and LLM evaluations. To test the effectiveness of the generated dataset, we train a multilingual Transformer classifier on the data in different training settings, comparing the results to a baseline model trained on a dataset, containing real cyber grooming chats. Additionally, to avoid overestimating the model’s performance, we utilize the CheckList framework,
a task agnostic methodology for testing models linguistic capabilities. 

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
### Chat generation
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

### Chat evaluation

To evaluate the chats using LLMs for the grammatical correctness attribute, use: 
```
python grammatical_correctness.py --model <model_name> --chats_file <chats_file_path.csv>
```

To evaluate the chats using LLMs for the coherence and overall quality attribute, use: 
```
python coherence_and_quality.py --model <model_name> --chats_file <chats_file_path.csv>
```

To evaluate the chats using LLMs for the prompt relevance attribute, use: 
```
python prompt_relevance.py --model <model_name> --chats_file <chats_file_path.csv>
```

For training XLM-R on generated data: 

```
python main.py --train
```

For evaluating the trained classification model: 
```
python main.py --evaluate --method --all
python main.py --evaluate --method --f1
python main.py --evaluate --method --classification_report
python main.py --evaluate --method --confusion_matrix
python main.py --evaluate --method --roc

```


