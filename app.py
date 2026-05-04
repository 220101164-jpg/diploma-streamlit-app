import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# LOAD DATA
df = pd.read_csv("diplomkaa.csv")

country_data = df.set_index("Country")

# TRAIN MODEL
X = df.drop(columns=["Country", "LE_index"])
y = df["LE_index"]

model = LinearRegression()
model.fit(X, y)

# TITLE
st.title("Life Expectancy Calculator")

# INPUTS
country = st.selectbox(
    "Select country",
    sorted(df["Country"].unique())
)

smoke = st.selectbox(
    "Smoking",
    [0, 1],
    format_func=lambda x: "No" if x == 0 else "Yes"
)

drink = st.selectbox(
    "Alcohol",
    [0, 1],
    format_func=lambda x: "No" if x == 0 else "Yes"
)

obesity = st.selectbox(
    "Obesity",
    [0, 1],
    format_func=lambda x: "No" if x == 0 else "Yes"
)

# BUTTON
if st.button("Predict"):

    data = country_data.loc[country].copy()

    real_le = data["LE_index"]

    data = data.drop("LE_index")

    # USER INPUTS
    data["smoking"] = smoke
    data["alcohol"] = drink
    data["Obesity"] = obesity

    # PREDICTION
    prediction = model.predict([data])[0]

    # RESULTS
    st.success(
        f"Predicted life expectancy: {prediction:.1f} years"
    )

    st.info(
        f"Estimated range: {prediction-3:.1f} - {prediction+3:.1f} years"
    )

    st.write(
        f"Average life expectancy in {country}: {real_le:.1f}"
    )
