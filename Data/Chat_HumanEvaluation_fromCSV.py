import csv
import os

# Define the file path for the input CSV file (containing chats)
input_csv_file_path = r'C:\Users\49171\Desktop\Conversation_evaluation\generated_chats_gpt3.5_batch2(85).csv'

# Define the file path for the output CSV file (containing evaluations)
output_csv_file_path = r'C:\Users\49171\Desktop\Conversation_evaluation\human_evaluations_batch85.csv'

# Check if the output CSV file already exists
file_exists = os.path.exists(output_csv_file_path)

# Open the input CSV file in read mode
with open(input_csv_file_path, 'r', newline='', encoding='utf-8') as csvfile:
    # Read conversations from the 'Chats' column
    reader = csv.DictReader(csvfile)
    conversations = [row['Chats'] for row in reader]

# Open the output CSV file in append mode
with open(output_csv_file_path, 'a', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['Chats', 'Grammatical Correctness', 'Coherence, Clarity, Quality', 'Prompt Relevance']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    # Write header only if the file is newly created
    if not file_exists:
        writer.writeheader()

    # Iterate through conversations for evaluation
    for i, conversation in enumerate(conversations, start=1):
        # Display the conversation to a human for evaluation
        print(f"\nConversation {i}:{conversation}")

        # provide evaluations for the following categories
        grammatical_correctness = int(input("Rate Grammatical Correctness (1-5): "))
        coherence_clarity = int(input("Rate Coherence, Clarity, and overall Quality (1-5): "))
        overall_quality = int(input("Rate Relevance to Prompt (1-5): "))

        # Write the evaluation for the current conversation
        writer.writerow({
            'Chats': conversation,
            'Grammatical Correctness': grammatical_correctness,
            'Coherence, Clarity, Quality': coherence_clarity,
            'Prompt Relevance': overall_quality,
        })

        print(f'Evaluation for Conversation {i} saved.')

print(f'Done. All evaluations saved to {output_csv_file_path}.')
