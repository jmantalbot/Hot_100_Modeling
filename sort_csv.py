import csv
from datetime import datetime

# Sorted by year and then by artist to make repairs easier. Manually fixed the last few years and brought down error rate considerably.
def sort_hot_100():
    input_filename = 'HOT 100_with_genres.csv'
    output_filename = 'sorted_hot_100.csv'

    try:
        with open(input_filename, mode='r', encoding='utf-8', errors='replace') as infile:
            reader = csv.DictReader(infile)
            fieldnames = reader.fieldnames
            data = list(reader)
        
        data.sort(key=lambda x: (
            -int(x['chart_date'][:4]) if x['chart_date'] and x['chart_date'][:4].isdigit() else 0, 
            x['performer'].lower() if x['performer'] else ""                                      
        ))

        with open(output_filename, mode='w', newline='', encoding='utf-8') as outfile:
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)

        print(f"Successfully sorted {len(data)} rows.")
        print(f"1. Sorted by Year (Most recent first)")
        print(f"2. Sorted by Artist (A-Z)")
        print(f"Output saved to: {output_filename}")

    except FileNotFoundError:
        print(f"Error: The file '{input_filename}' was not found. Please make sure it is in the same directory.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    sort_hot_100()