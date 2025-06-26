import csv

input_file = 'raw_reviews.csv'  # Your messy file
output_file = 'cleaned_reviews.csv'

with open(input_file, 'r', encoding='utf-8') as infile, \
     open(output_file, 'w', newline='', encoding='utf-8') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile, quoting=csv.QUOTE_ALL)
    writer.writerow(['user_id', 'review', 'timestamp'])  # Write header

    for row in reader:
        # Skip empty lines
        if not row or all(not cell.strip() for cell in row):
            continue

        # Remove extra columns, keep only first 3
        cleaned_row = row[:3]

        # Merge split lines if review is broken into multiple columns
        if len(cleaned_row) < 3:
            # Try to merge next line or columns if possible (manual fix may be needed for very messy data)
            continue  # Or handle as needed

        # Strip whitespace
        cleaned_row = [cell.strip() for cell in cleaned_row]

        writer.writerow(cleaned_row)

print('Cleaned data saved to cleaned_reviews.csv')