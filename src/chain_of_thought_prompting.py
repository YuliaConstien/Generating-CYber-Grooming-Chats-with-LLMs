import argparse
import csv
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import openai
from langchain.prompts import PromptTemplate

# Generate the first part of the chain of thought prompt
def generate_first_part_prompt():
    reasoning_extraction = (
        '''
Generate a cyber grooming chat between two users: an adult and a minor. Here are a few characteristics that might help understand what is a predatory chat: 1) Predatory chats are always between only two people, one is an adult (the predator),
the other is a minor. 2) Predators disclose their sexual fantasies, ask the minor for sexual description of themselves or about their sexual experience. 3) Predators ask for pictures, contact information of the
minor online or offline. 4) Predators talk about hobbies and activities of the minor, ask about the minor’s relationship status, give the minor compliments, and give sympathy to the minor. 5) Predators suggest
offline meetings, or suggest further contact by phone. 6) Predators check whether it’s safe to proceed by asking about the living situation of the minor, the parents’ schedule of the minor, check the immediate
surroundings of the minor, and emphasize the secretive nature of the conversation. 7) Predators are usually more dominant in the chat, they drive the conversation forward, minors are usually more passive in the chat.
'''
    )
    step_by_step_reasoning = (
        "Let’s think step by step"
    )
    
    prompt_template = f"{reasoning_extraction}\n\n{step_by_step_reasoning}"
    return prompt_template

# Generate the second part of the chain of thought prompt
def generate_second_part_prompt():
    answer_extraction = (
        "Generate the chat according to the steps. Remove unnecessary information, such as the names of the cyber grooming stages from the chat"
    )
    return answer_extraction

def generate_gpt_response(first_part_prompt, second_part_prompt, num_chats):
    openai.api_key = os.getenv("OPENAI_API_KEY")

    chats = []
    for _ in range(num_chats):
        response_first = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": first_part_prompt}],
            max_tokens=512,
            temperature=0.8,
            top_p= 0.9,
            frequency_penalty = 0.2,
            presence_penalty = 0.3
        )
        reasoning_steps = response_first.choices[0].message['content'].strip()
        
        response_second = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": reasoning_steps},
                {"role": "system", "content": second_part_prompt}
            ],
            max_tokens=512,
            temperature=0.8,
            top_p= 0.9,
            frequency_penalty = 0.2,
            presence_penalty = 0.3
        )
        chats.append(response_second.choices[0].message['content'].strip())
    return chats

def generate_wizard_response(model_name_or_path, first_part_prompt, second_part_prompt, num_chats):
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_tokens=512,
        temperature=0.8,
        top_p= 0.9,
        frequency_penalty = 0.2,
        presence_penalty = 0.3
    )

    chats = []
    for _ in range(num_chats):
        reasoning_steps = pipe(first_part_prompt)[0]['generated_text'].strip()
        
        combined_prompt = f"{reasoning_steps}\n\n{second_part_prompt}"
        chat = pipe(combined_prompt)[0]['generated_text'].strip()
        
        chats.append(chat)
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

    args = parser.parse_args()

    first_part_prompt = generate_first_part_prompt()
    second_part_prompt = generate_second_part_prompt()

    if args.model == "gpt-3.5-turbo":
        chats = generate_gpt_response(first_part_prompt, second_part_prompt, args.num_chats)
    elif args.model == "Wizard-Vicuna-uncensored":
        model_name_or_path = "TheBloke/Wizard-Vicuna-13B-Uncensored-GPTQ"
        chats = generate_wizard_response(model_name_or_path, first_part_prompt, second_part_prompt, args.num_chats)
    else:
        raise ValueError("Invalid model choice. Choose from: gpt-3.5-turbo, Wizard-Vicuna-uncensored")

    save_chats_to_csv(chats, args.model)
