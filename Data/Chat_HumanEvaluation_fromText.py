import csv
import os

# Define the file path for the conversations text file
conversations_file_path = r'C:\Users\49171\Desktop\Conversation_evaluation\Chats_uncencoredModel_text_200.txt'

# Read conversations from the text file
with open(conversations_file_path, 'r', encoding='utf-8') as conversations_file:
    conversations = conversations_file.read().split('\n\n\n')

# List to store evaluations
evaluations = []

# CSV file path
csv_file_path = 'Human_Eval_uncencoredModel_2.csv'

# Check if the CSV file already exists
file_exists = os.path.exists(csv_file_path)

# Open the CSV file in append mode
with open(csv_file_path, 'a', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['Chats', 'Grammatical Correctness', 'Coherence, Clarity, Quality','Prompt Relevance']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    # Write header only if the file is newly created
    if not file_exists:
        writer.writeheader()

    # Iterate through conversations for evaluation
    for i, conversation in enumerate(conversations, start=1):
        # Display the conversation to a human for evaluation
        print(f"\nConversation {i}:\n\n\n{conversation}")

        # provide evaluations for the following categories
        grammatical_correctness = int(input("Rate Grammatical Correctness (1-5): "))
        coherence_clarity = int(input("Rate Coherence, Clarity, and overall Quality (1-5): "))
        overall_quality = int(input("Rate Relevance to Prompt (1-5): "))

        # Save evaluations for the conversation
        evaluations.append({
            'Chat': conversation,
            'Grammatical Correctness': grammatical_correctness,
            'Coherence, Clarity, Quality': coherence_clarity,
            'Prompt Relevance': overall_quality,
            
        })

        # Write the evaluation for the current conversation
        writer.writerow({
            'Chats': conversation,
            'Grammatical Correctness': grammatical_correctness,
            'Coherence, Clarity, Quality': coherence_clarity,
            'Prompt Relevance': overall_quality,
            
        })

    print(f'All evaluations saved to {csv_file_path}.')

