import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import time
import os
import csv

# --- CONFIGURATION ---
SPOTIFY_CLIENT_ID = "client_id"
SPOTIFY_CLIENT_SECRET = "secret_id"


INPUT_FILE = 'HOT_100.csv'
OUTPUT_FILE = 'hot100.csv'

# Set to True to allow manual typing when not found
INTERACTIVE_MODE = False

# Set to True to scan existing file and fix "Error" rows using known artists
REPAIR_MODE = True

# --- GENRE MAPPING RULES ---
GENRE_MAPPING = {
    "Hip Hop": ["rap", "hip hop", "trap", "drill", "grime", "bounce", "urban"],
    "Rock": ["rock", "punk", "metal", "grunge", "new wave", "psychedelic"],
    "Country": ["country", "bluegrass", "americana", "folk"],
    "R&B/Soul": ["r&b", "soul", "disco", "funk", "gospel", "motown", "doo-wop", "new jack swing"],
    "Electronic": ["house", "edm", "techno", "electronica", "dance", "dubstep", "lo-fi"],
    "Latin": ["latin", "reggaeton", "salsa", "cumbia", "mariachi"],
    "Jazz/Blues": ["jazz", "blues"],
    "Pop": ["pop", "adult standards", "oldies", "schlager"] 
}

COMMON_GENRES = list(GENRE_MAPPING.keys()) + ["Alternative", "Indie", "Reggae", "Ska"]

# Initialize Spotify API
auth_manager = SpotifyClientCredentials(client_id=SPOTIFY_CLIENT_ID, client_secret=SPOTIFY_CLIENT_SECRET)
sp = spotipy.Spotify(auth_manager=auth_manager)


## clean up song names and artist names
def clean_text(text):
    if not isinstance(text, str):
        return str(text)
    text = text.replace('"', '').replace("'", "")
    text = " ".join(text.split())
    return text


# spotify has a lot of really strange genres, I'm trying to lump them into some broad categories for better analysis. This is probably the most subjective part of what I've done
# Is folk music really country? I'm not so sure, but Folk is a small enough genre that it's not super useful on its own.
def map_genres_to_broad_category(genre_list):
    for broad_category, keywords in GENRE_MAPPING.items():
        for genre in genre_list:
            for keyword in keywords:
                if keyword in genre.lower():
                    return broad_category
    if genre_list:
        return genre_list[0].title()
    return "No Genre Listed"


# Use the spotify api to get genre
# I ran into rate limits all the time doing this
# I made some fixes. If we've seen the song already, we can just cache it and use it later
# If we've seen the artist already, we can also cache it and use it later, instead of wasting precious api request.
# this has a big problem though. Artists like MGK and Post Malone that have changed genres might be recorded incorrectly. this should be a very small percentage of the data. Artists don't usually change genre.
def get_genre(song_name, artist_name):
    try:
        clean_song = clean_text(song_name)
        clean_artist = clean_text(artist_name)
        
        query = f"track:{clean_song} artist:{clean_artist}"
        results = sp.search(q=query, type='track', limit=1)

        items = results['tracks']['items']
        
        if not items:
            query = f"{clean_song} {clean_artist}"
            results = sp.search(q=query, type='track', limit=1)
            items = results['tracks']['items']
            
        if not items:
            return "Not Found"
            
        artist_id = items[0]['artists'][0]['id']
        artist = sp.artist(artist_id)
        genres = artist['genres']

        if genres:
            return map_genres_to_broad_category(genres)
        else:
            return "No Genre Listed"
            
    except Exception as e:
        print(f"Error fetching {song_name}: {e}")
        return "Error"


# Cache artists
def build_cache_from_dataframe(df):
    """
    Scans a dataframe and builds a dictionary of {Artist: Genre}.
    Crucially, it only remembers POSITIVE results.
    If the history says 'No Genre Listed', we ignore it so we can check again.
    """
    cache = {}
    if 'performer' not in df.columns or 'spotify_genre' not in df.columns:
        return cache
        
    for index, row in df.iterrows():
        artist = str(row['performer']).strip()
        genre = str(row['spotify_genre']).strip()
        
        invalid_markers = ["Error", "nan", "None", "Not Found", "No Genre Listed", ""]
        
        if genre not in invalid_markers:
            cache[artist] = genre
    return cache


## Main loop
def main():
    # 1. Load Data
    try:
        input_df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"Could not find {INPUT_FILE}.")
        return

    start_index = 0
    existing_df = None
    artist_genre_cache = {}

    # 2. Load Existing Progress & REPAIR
    if os.path.exists(OUTPUT_FILE):
        try:
            print("Reading existing output file...")
            # Read all columns as string initially to avoid type errors
            existing_df = pd.read_csv(OUTPUT_FILE, dtype=str)
            
            # Learn from existing valid entries
            artist_genre_cache = build_cache_from_dataframe(existing_df)
            print(f"Loaded knowledge base: {len(artist_genre_cache)} valid artist genres known.")

            if REPAIR_MODE:
                print("Running Repair Pass on existing rows...")
                updates_made = 0
                for i, row in existing_df.iterrows():
                    current_genre = str(row['spotify_genre'])
                    artist = str(row['performer']).strip()
                    
                    # If this row is bad/missing, but we know the artist from elsewhere
                    if current_genre in ["Error", "nan", "None", "Not Found", "No Genre Listed"] and artist in artist_genre_cache:
                        new_genre = artist_genre_cache[artist]
                        existing_df.at[i, 'spotify_genre'] = new_genre
                        updates_made += 1
                
                if updates_made > 0:
                    print(f"Repair complete! Fixed {updates_made} rows using historical data.")
                    # Overwrite the file with fixes
                    existing_df.to_csv(OUTPUT_FILE, index=False, quoting=csv.QUOTE_NONNUMERIC)
                else:
                    print("No repairable errors found.")

            start_index = len(existing_df)
            print(f"Resuming processing from row {start_index}...")

        except Exception as e:
            print(f"Error reading existing file: {e}. Starting fresh.")
            start_index = 0
    
    # 3. Initialize File if Fresh
    if start_index == 0:
        headers = list(input_df.columns) + ['spotify_genre']
        pd.DataFrame(columns=headers).to_csv(OUTPUT_FILE, index=False, quoting=csv.QUOTE_NONNUMERIC)
        print("Starting new processing run...")

    # 4. Main Processing Loop
    total_rows = len(input_df)
    
    # Session cache (Song + Artist) - remembers results just for this run
    session_cache = {}
    
    for i in range(start_index, total_rows):
        row = input_df.iloc[i].copy()
        
        song_raw = row['song']
        performer_raw = row['performer']
        
        clean_s = clean_text(song_raw)
        clean_p = clean_text(performer_raw)
        
        row['song'] = clean_s
        row['performer'] = clean_p
        
        # Clean 'previous_week'
        if 'previous_week' in row and pd.notnull(row['previous_week']) and row['previous_week'] != "":
            try:
                row['previous_week'] = int(float(row['previous_week']))
            except:
                pass 

        print(f"[{i+1}/{total_rows}] Processing: {clean_s} by {clean_p}")

        genre = None
        session_key = (clean_s, clean_p)

        # A. Check Session Cache (Specific song seen in this run)
        if session_key in session_cache:
            genre = session_cache[session_key]
            print(f"   -> Found in session cache: {genre}")

        # B. Check Historical Cache (Artist seen anywhere in file with a VALID genre)
        elif clean_p in artist_genre_cache:
            genre = artist_genre_cache[clean_p]
            print(f"   -> Found in history: {genre}")
            
        else:
            # C. Fetch from Spotify (Because it wasn't in cache or was "No Genre Listed" before)
            attempts = 0
            while genre is None and attempts < 3:
                try:
                    genre_result = get_genre(clean_s, clean_p)
                    if genre_result != "Error":
                        genre = genre_result
                    else:
                        attempts += 1
                        time.sleep(2)
                except:
                    attempts += 1
                    time.sleep(2)
            
            if genre is None:
                genre = "Error"

            # D. Manual Override
            if INTERACTIVE_MODE and genre in ["Not Found", "No Genre Listed", "Error"]:
                print("\n" + "!"*40)
                print(f"MISSING GENRE FOR: '{clean_s}' by '{clean_p}'")
                print(f"System returned: {genre}")
                print("Type a genre manually or press ENTER to skip.")
                
                valid_input = False
                while not valid_input:
                    user_input = input("Manual Genre > ").strip()
                    if user_input == "?":
                        print(", ".join(GENRE_MAPPING.keys()))
                    elif user_input.lower().startswith("search "):
                        term = user_input[7:].strip().lower()
                        matches = [g for g in COMMON_GENRES if term in g.lower()]
                        print(f"Matches: {', '.join(matches)}")
                    else:
                        if user_input:
                            genre = user_input
                        valid_input = True 

                print("!"*40 + "\n")
            
            # Update caches
            session_cache[session_key] = genre
            
            # We add to historical cache for this run only if it's a valid genre
            if genre not in ["Error", "Not Found", "No Genre Listed"]:
                artist_genre_cache[clean_p] = genre

        row['spotify_genre'] = genre
        
        pd.DataFrame([row]).to_csv(
            OUTPUT_FILE, 
            mode='a', 
            header=False, 
            index=False, 
            quoting=csv.QUOTE_NONNUMERIC
        )
        
        # Only sleep if we actually made a request (wasn't cached)
        if session_key not in session_cache and clean_p not in artist_genre_cache:
             time.sleep(10)

    print("Processing complete!")

if __name__ == "__main__":
      main()
