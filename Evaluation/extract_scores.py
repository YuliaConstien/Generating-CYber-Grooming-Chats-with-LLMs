import re
import csv
import argparse

def clean_and_extract_score(text):
    # Remove occurrences of "1-5", "out of 5", and "/5"
    text = text.replace("1-5", "").replace("out of 5", "").replace("/5", "")
    
    # Use regular expressions to find the first numerical value
    match = re.search(r'\b\d\b', text)
    if match:
        return int(match.group())
    else:
        raise ValueError("No numerical score found in text")

def process_scores(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        fieldnames = reader.fieldnames
        rows = list(reader)
        
    # Process each row to clean and extract the score
    for row in rows:
        for field in fieldnames[1:]:  # Skip the 'Chat' column
            try:
                row[field] = clean_and_extract_score(row[field])
            except ValueError:
                row[field] = None  # Set to None if no valid score is found

    # Save the cleaned scores back to a new CSV file
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and clean evaluation scores from LLM output.")
    parser.add_argument("--input_file", type=str, help="CSV file containing the raw evaluation scores", required=True)
    parser.add_argument("--output_file", type=str, help="CSV file to save the cleaned scores", required=True)
    
    args = parser.parse_args()
    
    process_scores(args.input_file, args.output_file)
