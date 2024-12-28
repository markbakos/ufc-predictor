import pandas as pd
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)

    df['event_date'] = pd.to_datetime(df['event_date'])
    df['fighter1_dob'] = pd.to_datetime(df['fighter1_dob'])
    df['fighter2_dob'] = pd.to_datetime(df['fighter2_dob'])


    df = df[~df['outcome'].isin(['Draw', 'No contest'])]

    fights_list = []

    for _, fight in df.iterrows():
        fight1 = {
            'event_date': fight['event_date'],
            'weight_class': fight['weight_class'],

            'main_fighter': fight['fighter1'],
            'main_height': fight['fighter1_height'],
            'main_weight': fight['fighter1_curr_weight'],
            'main_reach': fight['fighter1_reach'],
            'main_stance': fight['fighter1_stance'],
            'main_strikes_landed_pm': fight['fighter1_sig_strikes_landed_pm'],
            'main_strikes_accuracy': fight['fighter1_sig_strikes_accuracy'],
            'main_strikes_absorbed_pm': fight['fighter1_sig_strikes_absorbed_pm'],
            'main_strikes_defended': fight['fighter1_sig_strikes_defended'],
            'main_takedown_avg': fight['fighter1_takedown_avg_per15m'],
            'main_takedown_accuracy': fight['fighter1_takedown_accuracy'],
            'main_takedown_defence': fight['fighter1_takedown_defence'],
            'main_submission_attempts': fight['fighter1_submission_avg_attempted_per15m'],
            'main_age': (fight['event_date'] - pd.to_datetime(fight['fighter1_dob'])).days / 365.25,

            'opponent': fight['fighter2'],
            'opponent_height': fight['fighter2_height'],
            'opponent_weight': fight['fighter2_curr_weight'],
            'opponent_reach': fight['fighter2_reach'],
            'opponent_stance': fight['fighter2_stance'],
            'opponent_strikes_landed_pm': fight['fighter2_sig_strikes_landed_pm'],
            'opponent_strikes_accuracy': fight['fighter2_sig_strikes_accuracy'],
            'opponent_strikes_absorbed_pm': fight['fighter2_sig_strikes_absorbed_pm'],
            'opponent_strikes_defended': fight['fighter2_sig_strikes_defended'],
            'opponent_takedown_avg': fight['fighter2_takedown_avg_per15m'],
            'opponent_takedown_accuracy': fight['fighter2_takedown_accuracy'],
            'opponent_takedown_defence': fight['fighter2_takedown_defence'],
            'opponent_submission_attempts': fight['fighter2_submission_avg_attempted_per15m'],
            'opponent_age': (fight['event_date'] - pd.to_datetime(fight['fighter2_dob'])).days / 365.25,

            'win': 1
        }

        fight2 = {
            'event_date': fight['event_date'],
            'weight_class': fight['weight_class'],

            'main_fighter': fight['fighter2'],
            'main_height': fight['fighter2_height'],
            'main_weight': fight['fighter2_curr_weight'],
            'main_reach': fight['fighter2_reach'],
            'main_stance': fight['fighter2_stance'],
            'main_strikes_landed_pm': fight['fighter2_sig_strikes_landed_pm'],
            'main_strikes_accuracy': fight['fighter2_sig_strikes_accuracy'],
            'main_strikes_absorbed_pm': fight['fighter2_sig_strikes_absorbed_pm'],
            'main_strikes_defended': fight['fighter2_sig_strikes_defended'],
            'main_takedown_avg': fight['fighter2_takedown_avg_per15m'],
            'main_takedown_accuracy': fight['fighter2_takedown_accuracy'],
            'main_takedown_defence': fight['fighter2_takedown_defence'],
            'main_submission_attempts': fight['fighter2_submission_avg_attempted_per15m'],
            'main_age': (fight['event_date'] - pd.to_datetime(fight['fighter2_dob'])).days / 365.25,

            'opponent': fight['fighter1'],
            'opponent_height': fight['fighter1_height'],
            'opponent_weight': fight['fighter1_curr_weight'],
            'opponent_reach': fight['fighter1_reach'],
            'opponent_stance': fight['fighter1_stance'],
            'opponent_strikes_landed_pm': fight['fighter1_sig_strikes_landed_pm'],
            'opponent_strikes_accuracy': fight['fighter1_sig_strikes_accuracy'],
            'opponent_strikes_absorbed_pm': fight['fighter1_sig_strikes_absorbed_pm'],
            'opponent_strikes_defended': fight['fighter1_sig_strikes_defended'],
            'opponent_takedown_avg': fight['fighter1_takedown_avg_per15m'],
            'opponent_takedown_accuracy': fight['fighter1_takedown_accuracy'],
            'opponent_takedown_defence': fight['fighter1_takedown_defence'],
            'opponent_submission_attempts': fight['fighter1_submission_avg_attempted_per15m'],
            'opponent_age': (fight['event_date'] - pd.to_datetime(fight['fighter1_dob'])).days / 365.25,

            'win': 0
        }

        fights_list.extend([fight1, fight2])


    fights_df = pd.DataFrame(fights_list)

    fights_df['height_advantage'] = fights_df['main_height'] - fights_df['opponent_height']
    fights_df['reach_advantage'] = fights_df['main_reach'] - fights_df['opponent_reach']
    fights_df['weight_advantage'] = fights_df['main_weight'] - fights_df['opponent_weight']

    le = LabelEncoder()
    fights_df['weight_class_encoded'] = le.fit_transform(fights_df['weight_class'])
    fights_df['main_stance_encoded'] = le.fit_transform(fights_df['main_stance'].fillna('Unknown'))
    fights_df['opponent_stance_encoded'] = le.fit_transform(fights_df['opponent_stance'].fillna('Unknown'))

    fights_df['strike_differential'] = (fights_df['main_strikes_landed_pm'] - fights_df['main_strikes_absorbed_pm']) - (fights_df['opponent_strikes_landed_pm'] - fights_df['opponent_strikes_absorbed_pm'])
    fights_df['striking_efficiency'] = (fights_df['main_strikes_accuracy'] * fights_df['main_strikes_defended']) - (fights_df['opponent_strikes_accuracy'] * fights_df['opponent_strikes_defended'])
    fights_df['grappling_advantage'] = (fights_df['main_takedown_accuracy'] * fights_df['main_takedown_defence']) - (fights_df['opponent_takedown_accuracy'] * fights_df['opponent_takedown_defence'])

    fights_df['aggressive_score'] = ((fights_df['main_strikes_landed_pm'] + fights_df['main_takedown_avg'] * 5 + fights_df['main_submission_attempts'] * 3) -
                                     (fights_df['opponent_strikes_landed_pm'] + fights_df['opponent_takedown_avg'] * 5 + fights_df['opponent_submission_attempts'] * 3))
    fights_df['defensive_score'] = (fights_df['main_strikes_defended'] + fights_df['main_takedown_defence']) - (fights_df['opponent_strikes_defended'] + fights_df['opponent_takedown_defence'])

    return fights_df


def prepare_model_data(df):

    features = [
        'main_age', 'opponent_age',
        'height_advantage', 'reach_advantage', 'weight_advantage',
        'main_strikes_landed_pm', 'opponent_strikes_landed_pm',
        'main_strikes_accuracy', 'opponent_strikes_accuracy',
        'main_strikes_absorbed_pm', 'opponent_strikes_absorbed_pm',
        'main_strikes_defended', 'opponent_strikes_defended',
        'main_takedown_avg', 'opponent_takedown_avg',
        'main_takedown_accuracy', 'opponent_takedown_accuracy',
        'main_takedown_defence', 'opponent_takedown_defence',
        'main_submission_attempts', 'opponent_submission_attempts',
        'weight_class_encoded',
        'main_stance_encoded', 'opponent_stance_encoded',
        'strike_differential',
        'striking_efficiency',
        'grappling_advantage',
        'aggressive_score',
        'defensive_score'
    ]

    df[features] = df[features].fillna(df[features].mean())

    x = df[features].values
    y = df['win'].values

    x_train, x_temp, y_train, y_temp = train_test_split(x,y,test_size=0.3, random_state=42)
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)

    return x_train, x_val, x_test, y_train, y_val, y_test, scaler, features

if __name__ == "__main__":
    fights_df = load_and_preprocess_data("data/complete_ufc_data.csv")

    x_train, x_val, x_test, y_train, y_val, y_test, scaler, features = prepare_model_data(fights_df)
    print("Training set shape: ", x_train.shape)
    print("Validation set shape: ", x_val.shape)
    print("Test set shape: ", x_test.shape)

