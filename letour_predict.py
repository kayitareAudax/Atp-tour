import joblib
def predict_rider_rank(rider_name, year=None, model_path='letour_ranks.pkl'):
    model=joblib.load(model_path)
    scaler=joblib.load('letour_scaler.pkl')
    features=joblib.load('letour_features.pkl')
