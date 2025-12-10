import csv

"""
As I manually edited artist genres, this code looks through the csv, sees that I've made changes, and fixes genres if I've added to an artist above. 
A few problems this might cause: 1) Artists like Machine Gun Kelly, Post Malone that have changed Genres over their history.
                                 2) This does not work for collaborations between artists.
"""
def repair_hot_100():
    input_filename = 'repaired_hot_100.csv'
    output_filename = 'repaired_hot_100_2.csv'

    try:
        with open(input_filename, mode='r', encoding='utf-8', errors='replace') as infile:
            reader = csv.DictReader(infile)
            fieldnames = reader.fieldnames
            data = list(reader)
        
        artist_genres = {}
        # Define what we consider "invalid" that needs fixing
        invalid_genres = {"Error", "No Genre Listed", ""} 
        
        fixed_count = 0

        for row in data:
            current_genre = row.get('spotify_genre', '').strip()
            performer = row.get('performer', '')

            if current_genre.lower() == 'corrido' or current_genre.lower() == 'corridos tumbados' or current_genre.lower() == 'sad sierreño':
                current_genre = 'Latin'
                row['spotify_genre'] = current_genre

            if current_genre not in invalid_genres:
                artist_genres[performer] = current_genre
            
            else:
                if performer in artist_genres:
                    row['spotify_genre'] = artist_genres[performer]
                    fixed_count += 1

        with open(output_filename, mode='w', newline='', encoding='utf-8') as outfile:
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)

        print(f"Successfully processed {len(data)} rows.")
        print(f"- Input: {input_filename}")
        print(f"- Converted 'corrido' to 'Latin'.")
        print(f"- Backfilled {fixed_count} rows using artist history.")
        print(f"Output saved to: {output_filename}")

    except FileNotFoundError:
        print(f"Error: The file '{input_filename}' was not found. Please ensure the sorted file exists.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    repair_hot_100()
