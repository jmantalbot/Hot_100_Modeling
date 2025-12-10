import csv


## Analyze the entire Dataset to find out the percentage of each genre.
## I used this to see how much of the data was missing, before and after my entensive repairs
def analyze_genre_stats():
    input_filename = 'repaired_hot_100.csv' 

    total_rows = 0
    error_count = 0
    no_genre_count = 0

    try:
        with open(input_filename, mode='r', encoding='utf-8', errors='replace') as infile:
            reader = csv.DictReader(infile)
            
            for row in reader:
                total_rows += 1
                genre = row.get('spotify_genre', '').strip()
                
                if genre == "Error":
                    error_count += 1
                elif genre == "No Genre Listed" or genre == "":
                    no_genre_count += 1

        if total_rows > 0:
            error_percentage = (error_count / total_rows) * 100
            no_genre_percentage = (no_genre_count / total_rows) * 100
            total_missing_percentage = ((error_count + no_genre_count) / total_rows) * 100

            print(f"Analysis of file: {input_filename}")
            print(f"Total Rows: {total_rows}")
            print(f"Rows with 'Error': {error_count} ({error_percentage:.2f}%)")
            print(f"Rows with 'No Genre Listed': {no_genre_count} ({no_genre_percentage:.2f}%)")
            print(f"Total Missing/Invalid: {error_count + no_genre_count} ({total_missing_percentage:.2f}%)")
        else:
            print("The file appears to be empty.")

    except FileNotFoundError:
        print(f"Error: The file '{input_filename}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    analyze_genre_stats()