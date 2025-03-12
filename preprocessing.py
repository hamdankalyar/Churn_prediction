# preprocessing.py
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

# Load preprocessing objects
def load_preprocessing_objects(base_path='./models/'):
    with open(f'{base_path}one_hot_encoder_geo.pkl', 'rb') as file:
        one_hot_encoder_geo = pickle.load(file)
    with open(f'{base_path}label_encoder_gender.pkl', 'rb') as file:
        label_encoder_gender = pickle.load(file)
    with open(f'{base_path}scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    return one_hot_encoder_geo, label_encoder_gender, scaler

# Preprocess input data
def preprocess_input(input_data, one_hot_encoder_geo, label_encoder_gender, scaler):
    input_df = pd.DataFrame([input_data])

    # One-Hot Encode Geography
    geo_encoded = one_hot_encoder_geo.transform(input_df[['Geography']]).toarray()
    geo_feature_names = one_hot_encoder_geo.get_feature_names_out(['Geography'])
    geo_encoded_df = pd.DataFrame(geo_encoded, columns=geo_feature_names)

    # Label Encode Gender
    input_df['Gender'] = label_encoder_gender.transform(input_df['Gender'])

    # Concatenate encoded geography features and drop original Geography column
    input_df_encoded = pd.concat([input_df.drop('Geography', axis=1), geo_encoded_df], axis=1)

    # Scale the numerical input data
    input_scaled = scaler.transform(input_df_encoded)

    return input_scaled