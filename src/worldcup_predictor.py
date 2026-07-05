from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path
import pickle
import warnings
from sklearn.preprocessing import LabelEncoder,StandardScaler
warnings.filterwarnings("ignore")
import os

BASE_DIR=Path(__file__).resolve().parent.parent
MODELS_DIR=BASE_DIR/"models"
DATA_DIR=BASE_DIR/ "data"
MODELS_DIR.mkdir(parents=True,exist_ok=True)
def load_data()->pd.DataFrame:
    """Load the football result data sets"""
    try:
        fp=DATA_DIR/ "world_cup"/ "results.csv"
        if fp.exists():
            data=pd.read_csv(fp)
            return data
        else:
            print("The file doesn't exist")
            return pd.DataFrame()
    except Exception as e:
        print("An error happened", e)
        raise e
# preprocess the data
def process_data(data:pd.DataFrame)->pd.DataFrame:
    if data.empty:
        print("No data loaded, returning empty DataFrame")
        return pd.DataFrame()

    df=pd.DataFrame()
    if "tournament" in data.columns:
        df["tournament_enc"]=LabelEncoder().fit_transform(data["tournament"])
    df["goal_diff"]=data["home_score"]-data["away_score"]
    scaled_data=pd.concat([data.drop(columns=["tournament"]),df],axis=1)
    scaled_data=scaled_data.dropna()
    return scaled_data

def train_data(data:pd.DataFrame):
    if data.empty:
        print("No data to train on")
        return None, {}

    X=data.drop(columns=["goal_diff"])
    y=data["goal_diff"]  # Changed from DataFrame to Series
    categorical_cols = ['date', 'home_team', 'away_team', 'city', 'country']
    for col in categorical_cols:
        if col in X.columns:
            X[col] = X[col].astype('category')
    X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=32,test_size=0.2)
    print(f"\n  Training samples: {len(X_train):,}")
    print(f"  Test samples:     {len(X_test):,}")

    model=xgb.XGBRegressor(n_estimators=100, max_depth=6,learning_rate=0.05, random_state=32,subsample=0.8, objective="reg:squarederror",enable_categorical=True)
    model.fit(X_train,y_train,eval_set=[(X_test,y_test)],verbose=False)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n" + "═" * 55)
    print("  📊 Model Evaluation on Test Set")
    print("═" * 55)
    print(f"  MAE:            {mae:.4f}")
    print(f"  MSE:            {mse:.4f}")
    print(f"  R2 Score:       {r2:.4f}")
    print("═" * 55)
    print("Feature importance")
    importance=pd.DataFrame({
        "feature":X.columns,
        "importance":model.feature_importances_
    }).sort_values("importance",ascending=False)
    for _, row in importance.head(15).iterrows():
        bar = "█" * int(row["importance"] * 200)
        print(f"    {row['feature']:<30s} {row['importance']:.4f} {bar}")
    return model,{
        "mae": mae, "mse": mse, "r2": r2,
    }
def save_artifacts(model, metrics):
    """save model and its related metrics scores using pickel"""
    if model is None:
        print("No model to save")
        return

    artifacts={
        "model":model,
        "metrics":metrics
    }
    path=MODELS_DIR / "wc_xgb_model.pkl"
    with open(path,"wb") as f:
        pickle.dump(artifacts,f)
    print(f"Model saved to {path}  ({os.path.getsize(path) / 1024:.1f} KB)")
def main():
    """Handle preprocessing, training and testing"""
    print("═" * 55)
    print("  ⚽ World Cup XGBoost Match Predictor")
    print("═" * 55)
    print("\n[1/5] Loading the football match data...")
    raw = load_data()
    if raw.empty:
        print("  ❌ No data loaded. Please ensure results.csv exists in data/world_cup/")
        return
    print(f"  Total matches loaded: {len(raw):,}")
    print("\n[2/5] Engineering features...")
    scaled_data=process_data(raw)
    if scaled_data.empty:
        print("  ❌ No data after processing. Check data structure.")
        return
    print(f"  Data after processing: {len(scaled_data):,} rows")

    print("\n[3/5] Training XGBoost model...")
    model,metrics=train_data(scaled_data)
    if model is None:
        print("  ❌ Model training failed.")
        return

    print("\n[4/5] Saving model artifacts...")
    save_artifacts(model, metrics)

    print("\n" + "═" * 55)
    print("  ✅ Done! Run `src/match_predict.py` to make predictions.")
    print("═" * 55)
if __name__=="__main__":
    main()
