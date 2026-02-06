import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Models
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Generate synthetic data (reuse logic from train_model.py)
def generate_data(n_samples=1000):
    np.random.seed(42)
    data = {
        'screen_time': np.random.normal(8, 3, n_samples),
        'social_media_time': np.random.normal(3, 1.5, n_samples),
        'gaming_time': np.random.normal(1.5, 1, n_samples),
        'notifications': np.random.normal(150, 50, n_samples),
        'phone_pickups': np.random.normal(70, 30, n_samples),
        'sleep_hours': np.random.normal(7, 1.5, n_samples),
        'age': np.random.randint(15, 60, n_samples),
        'anxiety_level': np.random.randint(1, 11, n_samples),
        'feel_anxious': np.random.randint(0, 4, n_samples),
        'interrupt_sleep': np.random.randint(0, 4, n_samples),
        'neglect_responsibilities': np.random.randint(0, 4, n_samples),
        'failed_reduce': np.random.randint(0, 4, n_samples)
    }
    df = pd.DataFrame(data)
    cols = ['screen_time', 'social_media_time', 'gaming_time', 'notifications', 'phone_pickups', 'sleep_hours']
    for col in cols:
        df[col] = df[col].clip(lower=0)
        
    def determine_risk(row):
        score = 0
        if row['screen_time'] > 8: score += 2
        if row['social_media_time'] > 3: score += 1
        if row['gaming_time'] > 3: score += 1
        if row['notifications'] > 200: score += 1
        if row['phone_pickups'] > 100: score += 1
        if row['sleep_hours'] < 6: score += 2
        if row['anxiety_level'] > 7: score += 1
        if row['feel_anxious'] >= 2: score += 2
        if row['interrupt_sleep'] >= 2: score += 2
        if row['neglect_responsibilities'] >= 2: score += 2
        if row['failed_reduce'] >= 2: score += 1
        
        if score < 5: return 0
        elif score < 10: return 1
        else: return 2

    df['risk_level'] = df.apply(determine_risk, axis=1)
    return df

def compare_models():
    print("Generating data...")
    df = generate_data(2000) # Use more data for specific comparison
    X = df.drop('risk_level', axis=1)
    y = df['risk_level']
    
    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "SVM": SVC(probability=True, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5)
    }
    
    results = []
    
    print(f"\n{'Model':<25} | {'Accuracy':<10} | {'F1 Score (Weighted)':<20}")
    print("-" * 60)
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        results.append((name, acc, f1))
        print(f"{name:<25} | {acc:.4f}     | {f1:.4f}")
        
    print("-" * 60)
    
    # Find best model
    best_model_name, best_acc, best_f1 = max(results, key=lambda x: x[1])
    print(f"\n🏆 Best Model: {best_model_name} (Accuracy: {best_acc:.2%})")
    
    return best_model_name

if __name__ == "__main__":
    compare_models()
