import numpy as np
import pandas as pd
from typing import Optional, Dict
from datetime import datetime

from sklearn.preprocessing import LabelEncoder


class FighterStats:

    def __init__(self, file_path: str):

        self.df = pd.read_csv(file_path)

        self.df['event_date'] = pd.to_datetime(self.df['event_date'])
        self.df['fighter1_dob'] = pd.to_datetime(self.df['fighter1_dob'])
        self.df['fighter2_dob'] = pd.to_datetime(self.df['fighter2_dob'])

        self.weight_class_encoder = LabelEncoder()
        self.stance_encoder = LabelEncoder()

        all_stances = pd.concat([
            self.df['fighter1_stance'],
            self.df['fighter2_stance']
        ]).fillna('Unknown').unique()
        self.stance_encoder.fit(all_stances)

    def get_fighter_stats(self, fighter_name: str) -> Optional[Dict]:

        fighter_data = None

        f1_data = self.df[self.df['fighter1'] == fighter_name].sort_values('event_date', ascending=False)
        if not f1_data.empty:
            fighter_data = {
                'height': f1_data.iloc[0]['fighter1_height'],
                'weight': f1_data.iloc[0]['fighter1_curr_weight'],
                'reach': f1_data.iloc[0]['fighter1_reach'],
                'stance': f1_data.iloc[0]['fighter1_stance'],
                'strikes_landed_pm': f1_data.iloc[0]['fighter1_sig_strikes_landed_pm'],
                'strikes_accuracy': f1_data.iloc[0]['fighter1_sig_strikes_accuracy'],
                'strikes_absorbed_pm': f1_data.iloc[0]['fighter1_sig_strikes_absorbed_pm'],
                'strikes_defended': f1_data.iloc[0]['fighter1_sig_strikes_defended'],
                'takedown_avg': f1_data.iloc[0]['fighter1_takedown_avg_per15m'],
                'takedown_accuracy': f1_data.iloc[0]['fighter1_takedown_accuracy'],
                'takedown_defence': f1_data.iloc[0]['fighter1_takedown_defence'],
                'submission_attempts': f1_data.iloc[0]['fighter1_submission_avg_attempted_per15m'],
                'dob': f1_data.iloc[0]['fighter1_dob'],
                'weight_class': f1_data.iloc[0]['weight_class']
            }
        else:
            f2_data = self.df[self.df['fighter2'] == fighter_name].sort_values('event_date', ascending=False)
            if not f2_data.empty:
                fighter_data = {
                    'height': f1_data.iloc[0]['fighter2_height'],
                    'weight': f1_data.iloc[0]['fighter2_curr_weight'],
                    'reach': f1_data.iloc[0]['fighter2_reach'],
                    'stance': f1_data.iloc[0]['fighter2_stance'],
                    'strikes_landed_pm': f1_data.iloc[0]['fighter2_sig_strikes_landed_pm'],
                    'strikes_accuracy': f1_data.iloc[0]['fighter2_sig_strikes_accuracy'],
                    'strikes_absorbed_pm': f1_data.iloc[0]['fighter2_sig_strikes_absorbed_pm'],
                    'strikes_defended': f1_data.iloc[0]['fighter2_sig_strikes_defended'],
                    'takedown_avg': f1_data.iloc[0]['fighter2_takedown_avg_per15m'],
                    'takedown_accuracy': f1_data.iloc[0]['fighter2_takedown_accuracy'],
                    'takedown_defence': f1_data.iloc[0]['fighter2_takedown_defence'],
                    'submission_attempts': f1_data.iloc[0]['fighter2_submission_avg_attempted_per15m'],
                    'dob': f1_data.iloc[0]['fighter2_dob'],
                    'weight_class': f1_data.iloc[0]['weight_class']
                }

        return fighter_data

    def prepare_prediction_data(self, fighter1_name: str, fighter2_name: str) -> Optional[np.ndarray]:

        fighter1_stats = self.get_fighter_stats(fighter1_name)
        fighter2_stats = self.get_fighter_stats(fighter2_name)

        if not fighter1_stats or not fighter2_stats:
            return None

        current_date = datetime.now()
        fighter1_age = (current_date - pd.to_datetime(fighter1_stats['dob'])).days / 365.25
        fighter2_age = (current_date - pd.to_datetime(fighter2_stats['dob'])).days / 365.25

        height_advantage = fighter1_stats['height'] - fighter2_stats['height']
        reach_advantage = fighter1_stats['reach'] - fighter2_stats['reach']
        weight_advantage = fighter1_stats['weight'] - fighter2_stats['weight']

        strike_differential = (fighter1_stats['strikes_landed_pm'] - fighter1_stats['strikes_absorbed_pm']) - (fighter2_stats['strikes_landed_pm'] - fighter2_stats['strikes_absorbed_pm'])

        striking_efficiency = (fighter1_stats['strikes_accuracy'] * fighter1_stats['strikes_defended']) - (fighter2_stats['strikes_accuracy'] * fighter2_stats['strikes_defended'])

        grappling_advantage = (fighter1_stats['takedown_accuracy'] * fighter1_stats['takedown_defence']) - (fighter2_stats['takedown_accuracy'] * fighter2_stats['takedown_defence'])

        aggressive_score = (fighter1_stats['strikes_landed_pm'] + fighter1_stats['takedown_avg'] * 5 + fighter1_stats['submission_attempts'] * 3)

        defensive_score = (fighter1_stats['strikes_defended'] + fighter1_stats['takedown_defence'])

        stance1_encoded = self.stance_encoder.transform([fighter1_stats['stance'] or 'Unknown'])[0]
        stance2_encoded = self.stance_encoder.transform([fighter2_stats['stance'] or 'Unknown'])[0]

        features = [
            fighter1_age, fighter2_age,
            height_advantage, reach_advantage, weight_advantage,
            fighter1_stats['strikes_landed_pm'], fighter2_stats['strikes_landed_pm'],
            fighter1_stats['strikes_accuracy'], fighter2_stats['strikes_accuracy'],
            fighter1_stats['strikes_absorbed_pm'], fighter2_stats['strikes_absorbed_pm'],
            fighter1_stats['strikes_defended'], fighter2_stats['strikes_defended'],
            fighter1_stats['takedown_avg'], fighter2_stats['takedown_avg'],
            fighter1_stats['takedown_accuracy'], fighter2_stats['takedown_accuracy'],
            fighter1_stats['takedown_defence'], fighter2_stats['takedown_defence'],
            fighter1_stats['submission_attempts'], fighter2_stats['submission_attempts'],
            0,
            stance1_encoded, stance2_encoded,
            strike_differential,
            striking_efficiency,
            grappling_advantage,
            aggressive_score,
            defensive_score
        ]

        return np.array(features).reshape(1, -1)


def predict_fight(model, scaler, fighter_stats: FighterStats, fighter1_name: str, fighter2_name: str) -> tuple[str, float]:

    prediction_data = fighter_stats.prepare_prediction_data(fighter1_name, fighter2_name)

    if prediction_data is None:
        raise ValueError("Could not find data for one or both fighters.")

    scaled_data = scaler.transform(prediction_data)

    prediction = model.predict(scaled_data)[0][0]

    winner = fighter1_name if prediction > 0.5 else fighter2_name
    confidence = prediction if prediction > 0.5 else 1 - prediction

    return winner, float(confidence)

if __name__ == "__main__":

    from main import load_and_preprocess_data, prepare_model_data
    from model import create_model, train_model

    fights_df = load_and_preprocess_data("../data/complete_ufc_data.csv")
    x_train, x_val, x_test, y_train, y_val, y_test, scaler, features = prepare_model_data(fights_df)

    model = create_model(input_dim=x_train.shape[1])
    history = train_model(model, x_train, y_train, x_val, y_val)

    fighter_stats = FighterStats("../data/complete_ufc_data.csv")

    fighter1 = "Tom Aspinall"
    fighter2 = "Jon Jones"

    try:
        winner, confidence = predict_fight(model, scaler, fighter_stats, fighter1, fighter2)
        print(f"Predicted Winner: {winner}")
        print(f"Confidence: {confidence*100:.2f}%")
    except ValueError as e:
        print(f"Error: {e}")