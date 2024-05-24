import csv

def csv_to_text(input_csv_file, output_text_file):
    with open(input_csv_file, 'r', newline='', encoding='utf-8') as csv_file:
        reader = csv.DictReader(csv_file)
        conversations = [row['chats'] for row in reader]

    #Append conversations to the existing text file
    with open(output_text_file, 'a', encoding='utf-8') as text_file:
         #Check if the file is not empty before adding a separator
        if text_file.tell() != 0:
            text_file.write('\n\n')
        text_file.write('\n\n'.join(conversations))
        

# Replace 'input.csv' and 'output.txt' with the actual file paths
csv_to_text(r'C:\Users\49171\Desktop\Conversation_evaluation\generated_chats_gpt3_12.csv', r'C:\Users\49171\Desktop\Conversation_evaluation\Chats_HumanEval_gpt3.txt')
print(f'done')


# Count the number of conversations in the text file
with open(r'C:\Users\49171\Desktop\Conversation_evaluation\Chats_HumanEval_gpt3.txt', 'r', encoding='utf-8') as text_file:
    num_conversations = text_file.read().count('\n\n\n')
    print(f'Total number of conversations in the text file: {num_conversations}')