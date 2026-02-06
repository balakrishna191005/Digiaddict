
# Digital Addiction Risk Predictor

**Author:** Yalamanchi Balakrishna
**Technologies:** Python, Streamlit, Scikit-learn, Pandas, NumPy
**Project Type:** Machine Learning, Web App

---
## Overview

The **Digital Addiction Risk Predictor** is a machine learning-based web application that assesses users’ digital device usage patterns to predict the risk of digital addiction. The system analyzes daily screen time, social media usage, gaming hours, notifications, phone pickups, sleep patterns, and anxiety scores to categorize users into **Low, Moderate, or High risk levels**.

The application provides **personalized recommendations** and **key contributing factors** to help users manage their digital habits.

---

## Features

* Predicts digital addiction risk using a **Gradient Boosting Classifier**.
* Provides a **risk probability score**.
* Highlights **key contributing factors** for each user.
* Gives **personalized recommendations** to improve digital wellbeing.
* Visualizes **usage comparison** with healthy limits and global averages.
* Offers **additional resources** for digital wellbeing and professional support.

---

## Installation

1. Clone the repository:

```bash
git clone <repo_url>
cd digital-addiction-risk-predictor
```

2. Create a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

1. Train the model (if `addiction_model.pkl` is not available):

```bash
python train_model.py
```

2. Run the Streamlit app:

```bash
streamlit run app.py
```

3. Open the provided URL in a browser (default: [http://localhost:8501](http://localhost:8501)).

4. Input your digital usage statistics and behavioral patterns to see **risk predictions and recommendations**.

---

## File Structure

* `app.py` – Main Streamlit application.
* `train_model.py` – Script to train the Gradient Boosting model and save the scaler/model.
* `compare_models.py` – Script to compare different ML models.
* `addiction_model.pkl` – Serialized trained model and scaler.
* `mobile_addiction_data.csv` – Dataset used for training.
* `requirements.txt` – Python dependencies.

---

## Data

The dataset includes:

* **Screen_Time, Social_Media_Usage, Gaming_Hours, Notifications_Per_Day**
* **Phone_Unlocks, Sleep_Hours, Age, Anxiety_Score**
* Risk factors like **feel_anxious, interrupt_sleep, neglect_responsibilities, failed_reduce**

Target variable: `risk_level` (0 = Low, 1 = Moderate, 2 = High)

---

## Model

* **Algorithm:** Gradient Boosting Classifier
* **Scaling:** StandardScaler for feature normalization
* **Evaluation:** Accuracy, F1-score, classification report
* Trained on synthetic and real-world dataset samples to improve performance.

---

## Future Enhancements

* Add **user authentication** for personal tracking.
* Implement **historical tracking and trends** for digital usage.
* Deploy the app to **cloud platforms** like Heroku or Streamlit Cloud.
* Integrate more **psychological and behavioral factors**.

---

## References / Resources

* [Forest App](https://www.forestapp.cc/) – Gamify focus time
* [Freedom](https://freedom.to/) – Block distracting websites/apps
* [Google Digital Wellbeing](https://wellbeing.google/) – Analyze personal usage
* [Find a Helpline](https://findahelpline.com/) – Mental health support

---

