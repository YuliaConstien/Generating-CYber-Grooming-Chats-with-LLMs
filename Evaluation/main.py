import argparse
import os
import subprocess

def generate_chat(method, model, num_chats, examples_file):
    if method == "few_shot_prompting":
        subprocess.run(["python", "few_shot_prompting.py", "--model", model, "--num_chats", str(num_chats), "--examples_file", examples_file])
    elif method == "chain_of_thought":
        subprocess.run(["python", "chain_of_thought.py", "--model", model, "--num_chats", str(num_chats)])
    elif method == "self_refine_prompting":
        subprocess.run(["python", "self_refine_prompting.py", "--model", model, "--num_chats", str(num_chats)])
    else:
        print("Invalid prompting method. Choose from: few_shot_prompting, chain_of_thought, self_refine_prompting")

if __name__ == "__main__":
    # Set your OpenAI API key as an environment variable
    os.environ["OPENAI_API_KEY"] = "your_api_key_here"

    parser = argparse.ArgumentParser(description="Generate chat using different prompting methods and language models.")
    parser.add_argument("--method", type=str, choices=["few_shot_prompting", "chain_of_thought", "self_refine_prompting"], help="Prompting method to use")
    parser.add_argument("--model", type=str, choices=["gpt-3.5-turbo", "Wizard-Vicuna-uncensored"], help="Language model to use")
    parser.add_argument("--num_chats", type=int, default=200, help="Number of chats to generate")
    parser.add_argument("--examples_file", type=str, default="few_shot_examples.txt", help="File containing few-shot examples")

    args = parser.parse_args()

    generate_chat(args.method, args.model, args.num_chats, args.examples_file)
