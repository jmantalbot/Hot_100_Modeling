import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, Flatten, LSTM, Reshape, MaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping

"""
Reproducability and Config
"""
os.environ['PYTHONHASHSEED'] = '0'
random.seed(13)
np.random.seed(13)
tf.random.set_seed(13)

# Directories
PLOT_DIR = Path('plots2')
PLOT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR = Path('logs2')
LOG_DIR.mkdir(parents=True, exist_ok=True)
PRED_DIR = Path('predictions')
PRED_DIR.mkdir(parents=True, exist_ok=True)

# Logging Constants
CSV_LOG_PATH = LOG_DIR / "mae_log_longevity.csv"



"""
Logging MAE results

INPUT:
    model_name: string,
    mae_string: string
    epochs: integer

RETURN:
    None
"""
def log_csv_mae(model_name: str,
                mae_str: str,
                epochs: int):
    file_exists = CSV_LOG_PATH.exists()
    with CSV_LOG_PATH.open('a', encoding='utf-8') as outf:
        if not file_exists:
            outf.write('model_name,hiveid,yid,epochs,mae\n')
        outf.write(f'{model_name},{epochs},{mae_str}\n')
        outf.flush()
    print(f"[INFO] Logged MAE to: {CSV_LOG_PATH}")



"""
Plots Ground Truth vs Predicted.
Sorts data by Ground Truth to make the plot readable (S-curve).

INPUT:
    title: string
    xlabel: string
    ylabel: string
    y_true: Ground Truth
    y_pred: Prediction
    model_stub: string,
"""
def plot_y_yhat(title: str,
                xlabel: str,
                ylabel: str,
                y_true,
                y_pred,
                model_stub: str,
                save_flag: bool = True):

    fig, ax = plt.subplots(figsize=(10, 6))
    try:
        # Sort values for a cleaner plot (Low longevity -> High longevity)
        sorted_indices = np.argsort(y_true)
        y_true_sorted = y_true[sorted_indices]
        y_pred_sorted = y_pred[sorted_indices]

        ax.plot(y_true_sorted, label='Actual Weeks', linewidth=2, color='black', alpha=0.8)
        ax.scatter(range(len(y_pred_sorted)), y_pred_sorted, label='Predicted Weeks', color='dodgerblue', alpha=0.5, s=10)
        
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)

        if save_flag:
            out_fp = PLOT_DIR / f"{model_stub}_longevity_validation.png"
            fig.savefig(out_fp, bbox_inches='tight', dpi=150)
            print(f"[INFO] Saved plot to: {out_fp}")
    finally:
        plt.close(fig)




"""
Main Function, predicts the longevity of each song in the Hot 100 latest (August 2025)
"""
def predict_longevity():

    """
    0. Data Processing and Pre-Processing
    """
    print("Loading data...")
    try:
        df = pd.read_csv('hot100.csv', low_memory=False)
    except FileNotFoundError:
        print("Error: 'hot100.csv' not found.")
        return

    # Robust Date Parsing
    df['chart_date'] = pd.to_datetime(df['chart_date'], errors='coerce')
    df = df.dropna(subset=['chart_date'])
    
    # Filter junk genres
    df = df[~df['spotify_genre'].isin(['Error', 'No Genre Listed'])]



    """"
    1. Feature Extraction, Artist history, genre averages, etc.
    Prepare to use models.
    Find active songs, and split data for training and testing
    """
    print("Engineering features (Artist History, Genre Averages)...")

    last_date = df['chart_date'].max()
    print(f"Latest chart date found: {last_date.date()}")

    # Group by Song to get unique song stats
    song_stats = df.groupby(['song', 'performer', 'spotify_genre']).agg({
        'time_on_chart': 'max',       
        'peak_position': 'min',       
        'chart_date': 'max'           
    }).reset_index()

    # distinguish between active and inactive songs
    finished_songs = song_stats[song_stats['chart_date'] < last_date].copy()
    active_songs = song_stats[song_stats['chart_date'] == last_date].copy()

    # Avg Longevity per Genre
    genre_lifespans = finished_songs.groupby('spotify_genre')['time_on_chart'].mean()
    global_avg_life = finished_songs['time_on_chart'].mean()

    # Avg Longevity per Artist
    artist_lifespans = finished_songs.groupby('performer')['time_on_chart'].mean()
    global_artist_avg = finished_songs['time_on_chart'].mean()


    # Helper songs to find genre and artist average.
    def get_genre_avg(genre):
        return genre_lifespans.get(genre, global_avg_life)

    def get_artist_avg(artist):
        return artist_lifespans.get(artist, global_artist_avg)


    # We only need to look at active songs to make our predictions
    for dataset in [finished_songs, active_songs]:
        dataset['genre_avg_life'] = dataset['spotify_genre'].apply(get_genre_avg)
        dataset['artist_avg_life'] = dataset['performer'].apply(get_artist_avg)
    
    feature_cols = ['peak_position', 'genre_avg_life', 'artist_avg_life']
    target_col = 'time_on_chart'

    X_train_raw = finished_songs[feature_cols].values
    y_train_raw = finished_songs[target_col].values
    
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train_raw)
    
    X_train, X_val, y_train, y_val = train_test_split(X_train_scaled, y_train_raw, test_size=0.2, random_state=13)
    n_features = X_train.shape[1]


    """
    2. Define models. Using ANN, CNN, LSTM
    """
    def build_ann():
        model = Sequential([
            Dense(64, activation='relu', input_shape=(n_features,)),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mae')
        return model

    def build_cnn():
        model = Sequential([
            Reshape((n_features, 1), input_shape=(n_features,)),
            Conv1D(32, kernel_size=2, activation='relu', padding='same'),
            MaxPooling1D(pool_size=2, padding='same'),
            Flatten(),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mae')
        return model

    def build_lstm():
        model = Sequential([
            Reshape((1, n_features), input_shape=(n_features,)),
            LSTM(64),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mae')
        return model

    models = {
        "ANN": build_ann(),
        "CNN": build_cnn(),
        "LSTM": build_lstm()
    }








    """
    3. Train and Predict
    """
    print(f"\nTraining models on {len(X_train)} finished songs...")
    
    X_active_raw = active_songs[feature_cols].values
    X_active_scaled = scaler_X.transform(X_active_raw)
    current_weeks_on_chart = active_songs['time_on_chart'].values

    final_results = active_songs[['song', 'performer', 'spotify_genre', 'time_on_chart']].copy()
    final_results.rename(columns={'time_on_chart': 'current_weeks'}, inplace=True)

    EPOCHS = 50

    for name, model in models.items():
        print(f"\n--- Processing {name} ---")
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        
        # Train
        model.fit(X_train, y_train, 
                  validation_data=(X_val, y_val),
                  epochs=EPOCHS, 
                  batch_size=32, 
                  callbacks=[early_stop],
                  verbose=0)
        
        # Evaluate
        loss = model.evaluate(X_val, y_val, verbose=0)
        print(f"{name} Validation MAE: +/- {loss:.2f} weeks")

        # Log Metrics
        log_csv_mae(
            model_name=name,
            mae_str=f"{loss:.5f}",
            epochs=EPOCHS
        )

        # Plot Validation Results
        val_preds = model.predict(X_val, verbose=0).flatten()
        plot_y_yhat(
            title=f"{name} Validation: Predicted vs Actual Weeks",
            xlabel="Samples (Sorted by Actual Duration)",
            ylabel="Total Weeks on Chart",
            y_true=y_val,
            y_pred=val_preds,
            model_stub=name.lower()
        )

        # Predict Active Songs
        preds = model.predict(X_active_scaled, verbose=0).flatten()
        
        # Calculate Remaining
        remaining = preds - current_weeks_on_chart
        remaining = np.maximum(remaining, 0)
        
        final_results[f'{name}_remaining'] = remaining






    """"
    4. Display Results
    """
    print("\n" + "="*80)
    print(f"PREDICTED REMAINING WEEKS FOR ALL SONGS ({last_date.date()})")
    print("="*80)
    
    # Save to CSV
    out_csv = PRED_DIR / "current_chart_predictions.csv"
    final_results.to_csv(out_csv, index=False)
    print(f"[INFO] Full predictions saved to: {out_csv}")
    print("-" * 80)

    # Sort by ANN prediction for display
    final_results = final_results.sort_values(by='ANN_remaining', ascending=False)
    
    # Print header
    print(f"{'Song':<30} | {'Artist':<20} | {'Cur':<3} | {'ANN':<5} | {'CNN':<5} | {'LSTM':<5}")
    print("-" * 80)

    # Set pandas option to ensure all rows print if user runs this locally
    pd.set_option('display.max_rows', None)

    # Iterate and print cleanly formatted rows
    for _, row in final_results.iterrows():
        song_title = (row['song'][:27] + '..') if len(str(row['song'])) > 27 else str(row['song'])
        artist_name = (row['performer'][:17] + '..') if len(str(row['performer'])) > 17 else str(row['performer'])
        
        print(f"{song_title:<30} | {artist_name:<20} | {int(row['current_weeks']):<3} | "
              f"{row['ANN_remaining']:<5.1f} | {row['CNN_remaining']:<5.1f} | {row['LSTM_remaining']:<5.1f}")

    print("-" * 80)
    print("Done.")

if __name__ == "__main__":
    predict_longevity()