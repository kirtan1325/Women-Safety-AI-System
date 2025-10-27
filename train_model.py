# train_model.py
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from joblib import dump

# Synthetic dataset generation
def generate_data(n=20000, random_state=42):
    rng = np.random.RandomState(random_state)
    # features:
    # hour(0-23), crime_rate(0-1), lighting(0-1), crowd_density(0-1), proximity_police_km (0-10), recent_reports(0-20)
    hour = rng.randint(0,24,size=n)
    crime_rate = rng.beta(2,6,size=n)  # many low, few high
    lighting = rng.beta(5,2,size=n)    # many well-lit
    crowd = rng.beta(2,3,size=n)
    prox_police = rng.exponential(scale=1.5,size=n)
    recent_reports = rng.poisson(1.0,size=n)

    # base safety on lighting & crime & crowd & time
    safety = 70*lighting - 60*crime_rate + 10*(1 - crowd) - 3*(np.clip(hour-20,0,6)) - 2*recent_reports \
             - 2*np.log1p(prox_police)
    # normalize to 0-100
    safety = (safety - safety.min()) / (safety.max() - safety.min()) * 100
    # add some noise
    safety = np.clip(safety + rng.normal(0,6,size=n), 0, 100)
    df = pd.DataFrame({
        'hour': hour,
        'crime_rate': crime_rate,
        'lighting': lighting,
        'crowd': crowd,
        'prox_police': prox_police,
        'recent_reports': recent_reports,
        'safety': safety
    })
    return df

def train_and_save():
    df = generate_data()
    X = df[['hour','crime_rate','lighting','crowd','prox_police','recent_reports']]
    y = df['safety']
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    print("Train score:", model.score(X_train, y_train))
    print("Test score:", model.score(X_test, y_test))
    dump(model, 'safety_model.joblib')
    print("Saved model to safety_model.joblib")

if __name__ == "__main__":
    train_and_save()
