import argparse
import csv
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import openai
from langchain.prompts import PromptTemplate

# Generate initial chat prompt
def generate_initial_prompt():
    return '''
Generate a cyber grooming chat between two users: an adult and a minor. Here are a few characteristics that might help understand what a predatory chat is: 1) Predatory chats are always between only two
people, one is an adult (the predator), the other is a minor. 2) Predators disclose their sexual fantasies, ask the minor for sexual description of themselves or about their sexual experience. 3) Predators ask for pic-
tures, contact information of the minor online or offline. 4) Predators talk about hobbies and activities of the minor, ask about the minor’s relationship status, give the minor compliments, and give sympathyto the minor. 5) Predators suggest offline meetings, or suggest further
contact by phone. 6) Predators check whether it’s safe to proceed by asking about the living situation of the minor, the parents’ schedule of the minor, check the immediate surroundings of the minor, and emphasize the secretive nature of the conversation. 7) Predators are usu-
ally more dominant in the chat, they drive the conversation forward, minors are usually more passive in the chat.
'''

# Generate feedback prompt template
def generate_feedback_prompt(chat):
    return (
        f"Chat:\n{chat}\n\n"
        "Give feedback for the generated chat. Specify what mistakes did you find in the chat and how to correct them. Consider the text’s grammar and punctuation, relevance to the task, coherence, transition between topics and overall quality."
    )

# Generate refinement prompt template
def generate_refinement_prompt(chat, feedback):
    return (
        f"Chat:\n{chat}\n\n"
        f"Feedback:\n{feedback}\n\n"
        "Consider the feedback above and make changes to the chat according to it. Consider all points of the feedback."
    )

def generate_gpt_response(initial_prompt, num_chats):
    openai.api_key = os.getenv("OPENAI_API_KEY")

    chats = []
    for _ in range(num_chats):
        # Generate initial chat
        response_initial = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": initial_prompt}],
             temperature=0.8,
            top_p= 0.9,
            frequency_penalty = 0.2,
            presence_penalty = 0.3
        )
        initial_chat = response_initial.choices[0].message['content'].strip()

        # Generate feedback
        feedback_prompt = generate_feedback_prompt(initial_chat)
        response_feedback = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": feedback_prompt}]
        )
        feedback = response_feedback.choices[0].message['content'].strip()

        # Refine chat based on feedback
        refinement_prompt = generate_refinement_prompt(initial_chat, feedback)
        response_refined = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": refinement_prompt}]
        )
        refined_chat = response_refined.choices[0].message['content'].strip()

        chats.append(refined_chat)
    return chats

def generate_wizard_response(model_name_or_path, initial_prompt, num_chats):
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
        # Generate initial chat
        initial_chat = pipe(initial_prompt)[0]['generated_text'].strip()

        # Generate feedback
        feedback_prompt = generate_feedback_prompt(initial_chat)
        feedback = pipe(feedback_prompt)[0]['generated_text'].strip()

        # Refine chat based on feedback
        refinement_prompt = generate_refinement_prompt(initial_chat, feedback)
        refined_chat = pipe(refinement_prompt)[0]['generated_text'].strip()

        chats.append(refined_chat)
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

    initial_prompt = generate_initial_prompt()

    if args.model == "gpt-3.5-turbo":
        chats = generate_gpt_response(initial_prompt, args.num_chats)
    elif args.model == "Wizard-Vicuna-uncensored":
        model_name_or_path = "TheBloke/Wizard-Vicuna-13B-Uncensored-GPTQ"
        chats = generate_wizard_response(model_name_or_path, initial_prompt, args.num_chats)
    else:
        raise ValueError("Invalid model choice. Choose from: gpt-3.5-turbo, Wizard-Vicuna-uncensored")

    save_chats_to_csv(chats, args.model)
