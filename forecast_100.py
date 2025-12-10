import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.callbacks import EarlyStopping

## For Reproducability
os.environ['PYTHONHASHSEED'] = '0'
random.seed(13)
np.random.seed(13)
tf.random.set_seed(13)


# Directories and Paths, Replace with your own paths
PLOT_DIR = Path('plots')
PLOT_DIR.mkdir(parents=True, exist_ok=True)
CSV_FILE_PATH_PAT = "logs/mse_log_in{}_out{}.csv"


"""
Logs model metadata and MSE results to a CSV file.

INPUT:
    model_name: string,
    in_steps: integer,
    out_steps: integer,
    mse_str: string,
    epochs, integer
    val_split: float

RETURN:
    None
"""
def log_csv_mse(model_name: str,
                in_steps: int,
                out_steps: int,
                mse_str: str,
                epochs: int,
                val_split: float):
    """
    """
    log_fp = Path(str(CSV_FILE_PATH_PAT).format(in_steps, out_steps))
    log_fp.parent.mkdir(parents=True, exist_ok=True)
    file_exists = log_fp.exists()
    with log_fp.open('a', encoding='utf-8') as outf:
        if not file_exists:
            outf.write('model_name,in_steps,out_steps,epochs,val_split,mse\n')
        outf.write(f'{model_name},{in_steps},{out_steps},{epochs},{val_split},{mse_str}\n')
        outf.flush()





"""
Generates a plot for a specific genre comparing True vs Predicted counts.

INPUT:
    genre_name: string,
    y_true: Ground truth data
    y_pred: Prediction data
    model_name: string,
    config_info: string

RETURN:
    NONE
"""
def plot_genre_specific(genre_name: str,
                        y_true,
                        y_pred,
                        model_name: str,
                        config_info: str):
    
    fig, ax = plt.subplots(figsize=(10, 6))
    try:
        ax.plot(y_true, label='Ground Truth', linewidth=2, color='black', alpha=0.7)
        ax.plot(y_pred, label='Prediction', linestyle='--', linewidth=2, color='dodgerblue', alpha=0.9)
        
        # set labels, titles, legend 
        title = f"{model_name} Best Fit ({config_info}) - {genre_name}"
        ax.set_title(title)
        ax.set_xlabel("Test Weeks")
        ax.set_ylabel("Song Count")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)

        # Output file
        safe_genre = genre_name.replace('/', '_').replace(' ', '_')
        out_fp = PLOT_DIR / f"Best_{model_name}_{safe_genre}.png"
        fig.savefig(out_fp, bbox_inches='tight', dpi=150)
    finally:
        plt.close(fig)






"""
Main function

INPUT:
    NONE

Returns:
    NONE
"""
def compare_genre_models():
    """
    0. DATA PROCESSING 
    
    Finds hot100.csv, coerces errors, filters Error and No Genre Listed
    Aggregates by Genre per week and Scales Data
    """
    print("Loading data...")
    try:
        # Check if CSV exists
        df = pd.read_csv('hot100.csv', low_memory=False)
    except FileNotFoundError:
        print("Error: 'hot100.csv' not found. Please ensure the file exists.")
        return

    # Coerce Data
    df['chart_date'] = pd.to_datetime(df['chart_date'], errors='coerce')
    df = df.dropna(subset=['chart_date'])

    # Filter unwanted genres
    invalid_genres = ['Error', 'No Genre Listed']
    df = df[~df['spotify_genre'].isin(invalid_genres)]

    # Aggregate by Genre per Week
    top_genres = df['spotify_genre'].value_counts().nlargest(10).index.tolist()
    df['clean_genre'] = df['spotify_genre'].apply(lambda x: x if x in top_genres else 'Other')
    
    pivot_df = df.pivot_table(index='chart_date', columns='clean_genre', values='song', aggfunc='count', fill_value=0)
    pivot_df = pivot_df.sort_index()
    
    # Scale Data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(pivot_df)
    




    """
    1. TRAIN TEST SPLIT
    Look at the last sixty weeks as a window
    test size is the last year.     
    """
    # Sliding Windows
    look_back = 60
    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i-look_back:i])
        y.append(scaled_data[i])
        
    X, y = np.array(X), np.array(y)
    
    # Train/Test Split
    test_size = 52
    X_train, X_test = X[:-test_size], X[-test_size:]
    y_train, y_test = y[:-test_size], y[-test_size:]
    
    n_features = X.shape[2]
    input_shape = (look_back, n_features)

    

    """
    2. Create Models for LSTM, CNN, and ANN, set up for val_splits, epochs, and models
    """
    val_splits = [0.1, 0.2]
    epochs_list = [5, 20, 100]
    model_names = ["LSTM", "CNN", "ANN"]

    # Dictionary to store the BEST results for each architecture
    # Structure: {'LSTM': {'score': 999, 'y_pred': ..., 'config': '...'}, ...}
    best_models = {}

    def get_fresh_model(name):
        if name == "LSTM":
            return Sequential([
                LSTM(64, return_sequences=False, input_shape=input_shape),
                Dropout(0.2), Dense(n_features)
            ])
        elif name == "CNN":
            return Sequential([
                Conv1D(64, 3, activation='relu', input_shape=input_shape),
                MaxPooling1D(2), Flatten(), Dense(50, activation='relu'), Dense(n_features)
            ])
        elif name == "ANN":
            return Sequential([
                Flatten(input_shape=input_shape),
                Dense(128, activation='relu'), Dropout(0.2), Dense(64, activation='relu'), Dense(n_features)
            ])
        return None

    print(f"Starting experiments across {len(val_splits) * len(epochs_list)} configurations per model...")


    """
    3. Begin Experiements. We stop early if not enough progress is being made. this never happened in my testing, but it can save us time if we really ramp up the number of epochs.
    """
    for val_split in val_splits:
        for num_epochs in epochs_list:
            for name in model_names:
                # 1. Build & Train
                model = get_fresh_model(name)
                model.compile(optimizer='adam', loss='mse')
                
                early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
                
                model.fit(X_train, y_train, epochs=num_epochs, batch_size=32, 
                          validation_split=val_split, callbacks=[early_stop], verbose=0)
                
                # 2. Evaluate
                loss = model.evaluate(X_test, y_test, verbose=0)
                y_pred_scaled = model.predict(X_test, verbose=0)

                # 3. Log
                log_csv_mse(name, look_back, 1, f"{loss:.5f}", num_epochs, val_split)
                
                # 4. Check if Best
                current_best = best_models.get(name, {'score': float('inf')})
                if loss < current_best['score']:
                    print(f"New Best {name}: MSE {loss:.5f} (Epochs={num_epochs}, Val={val_split})")
                    best_models[name] = {
                        'score': loss,
                        'y_pred_scaled': y_pred_scaled,
                        'config': f"e{num_epochs}_v{val_split}",
                        'model_obj': model # Save actual object for future forecast
                    }

  
    """
    4. PLOT The best models by Genre
    """
    print("\n" + "="*60)
    print("GENERATING PLOTS FOR BEST MODELS")
    print("="*60)

    # Inverse transform Ground Truth once (it's the same for all)
    y_true_real = scaler.inverse_transform(y_test)
    genres = pivot_df.columns

    for name, data in best_models.items():
        print(f"Plotting genres for Best {name} ({data['config']})...")
        
        # Inverse transform predictions
        y_pred_real = scaler.inverse_transform(data['y_pred_scaled'])
        
        # Clip negatives. We can't have -n songs in the Hot 100.
        y_pred_real = np.maximum(y_pred_real, 0)

        # Plot each genre separately
        for i, genre in enumerate(genres):
            plot_genre_specific(
                genre_name=genre,
                y_true=y_true_real[:, i],
                y_pred=y_pred_real[:, i],
                model_name=name,
                config_info=data['config']
            )

    """
    5. Forecast the next 52 weeks using the best model of each type
    """
    print("\n" + "="*60)
    print("FUTURE FORECAST (NEXT 52 WEEKS) USING WINNERS")
    print("="*60)

    for name, data in best_models.items():
        print(f"\n>>> {name} Forecast ({data['config']}) <<<")
        
        model = data['model_obj']
        current_batch = scaled_data[-look_back:].reshape(1, look_back, n_features)
        future_preds = []
        
        for _ in range(52):
            pred = model.predict(current_batch, verbose=0)
            future_preds.append(pred[0])
            current_batch = np.append(current_batch[:, 1:, :], [pred], axis=1)
            
        future_preds = scaler.inverse_transform(future_preds)
        avg_future = np.mean(future_preds, axis=0)
        avg_future = np.maximum(avg_future, 0)
        
        # Normalize to 100
        total = np.sum(avg_future)
        if total > 0:
            avg_future = (avg_future / total) * 100
            
        results = list(zip(genres, avg_future))
        results.sort(key=lambda x: x[1], reverse=True)
        
        for g, c in results:
            print(f"{g}: {c:.1f}")

    print("\nDone. Check 'plots/' for genre-specific graphs.")

if __name__ == "__main__":
    compare_genre_models()