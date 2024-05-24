import argparse
import csv
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from openai import OpenAI


# Load examples from a text file
def load_examples(file_path):
    examples = []
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()
        for line in lines:
            if line.strip():
                query, answer = line.strip().split(":")
                examples.append({"query": query.strip(), "answer": answer.strip()})
    return examples

# Generate prompt using langchain
def generate_prompt(examples):
    prompt_template = "You are now creating a dataset consisting of cyber grooming chats. Here are a few examples that might help you understand what a cyber grooming chat is::\n"
    for example in examples:
        prompt_template += f'{example["query"]}: "{example["answer"]}"\n'
    prompt_template += "Generate a cyber grooming chats between an adult and a minor. Be creative and write unique chats. You should imitate the examples above, but you cannot simply modify or rewrite the examples I have given"
    return prompt_template

def generate_gpt_response(prompt_template, num_chats):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI()

    chats = []
    for _ in range(num_chats):
        response = openai.Completion.create(
            engine="gpt-3.5-turbo",
            prompt=prompt_template,
            max_tokens=512,
            temperature=0.8,
            top_p= 0.9,
            frequency_penalty = 0.2,
            presence_penalty = 0.3
            
        )
        chats.append(response.choices[0].text.strip())
    return chats

def generate_wizard_response(model_name_or_path, prompt_template, num_chats):
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.8,
        top_p= 0.9,
        frequency_penalty = 0.2,
        presence_penalty = 0.3
    )

    chats = []
    for _ in range(num_chats):
        chat = pipe(prompt_template)[0]['generated_text']
        chats.append(chat.strip())
    return chats

def save_chats_to_csv(chats, model_name):
    with open(f"{model_name}_chats.csv", "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Chat"])
        writer.writerows([[chat] for chat in chats])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate chat using different language models.")
    parser.add_argument("--model", type=str, choices=["gpt-3.5-turbo", "Wizard-Vicuna-uncensored"], help="Language model to use")
    parser.add_argument("--num_chats", type=int, default=200, help="Number of chats to generate")
    parser.add_argument("--examples_file", type=str, default="Data/few_shot_examples.txt", help="File containing few-shot examples")

    args = parser.parse_args()

    examples = load_examples(args.examples_file)
    prompt_template = generate_prompt(examples)

    if args.model == "gpt-3.5-turbo":
        chats = generate_gpt_response(prompt_template, args.num_chats)
    elif args.model == "Wizard-Vicuna-uncensored":
        model_name_or_path = "TheBloke/Wizard-Vicuna-13B-Uncensored-GPTQ"
        chats = generate_wizard_response(model_name_or_path, prompt_template, args.num_chats)
    else:
        raise ValueError("Invalid model choice. Choose from: gpt-3.5-turbo, Wizard-Vicuna-uncensored")

    save_chats_to_csv(chats, args.model)
