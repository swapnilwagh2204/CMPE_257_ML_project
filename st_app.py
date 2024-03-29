import streamlit as st
import pickle
import pandas as pd


import wandb
import random

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="ccs - initial models",
    name="0.0.2",

    # track hyperparameters and run metadata
)

# Load pickled model and scaler
pickled_model = pickle.load(open('finalized_model.sav', 'rb'))
pickled_scaler = pickle.load(open('scaler.pkl', 'rb'))

# Define predict function


def predict(cement, blast_furnace_slag, fly_ash, water, superplasticizer, coarse_aggregate, fine_aggregate, age):

    # Transform input data using pickled scaler
    scaler_list = pickled_scaler.transform(
        [[cement, blast_furnace_slag, fly_ash, water, superplasticizer, coarse_aggregate, fine_aggregate, age]]).tolist()

    # Predict using pickled model
    prediction = pickled_model.predict(pd.DataFrame(scaler_list, columns=["cement", "blast_furnace_slag", "fly_ash", "water", "superplasticizer",
                                                                          "coarse_aggregate", "fine_aggregate", "age"]))
    return prediction[0]


# # Streamlit app code
# st.title('Compressive Strength Concrete Predictor')
# st.image('https://civildigital.com/wp-content/uploads/2016/07/Hydraulic-Compression-Testing-Machine.jpg')
# st.header('Enter the Components of Concrete')

# cement = st.number_input('cement:')
# blast_furnace_slag = st.number_input('blast_furnace_slag:')
# fly_ash = st.number_input('fly_ash:')
# water = st.number_input('water:')
# superplasticizer = st.number_input('superplasticizer:')
# coarse_aggregate = st.number_input('Coarse_aggregates:')
# fine_aggregate = st.number_input('Fine_aggregates:')
# age = st.slider('Age in years:')

# if st.button('Predict'):
#     price = predict(cement, blast_furnace_slag, fly_ash, water,
#                     superplasticizer, coarse_aggregate, fine_aggregate, age)
#     st.success(
#         f'The predicted compressive strength of concrete is {price:.2f} MPa.')

if __name__ == '__main__':
    price = predict(139.6, 209.4, 0.0, 192.0, 0.0, 1047.0, 806.9, 90)
    print("price - ", price)
    wandb.log({"price": price})

    wandb.finish()
