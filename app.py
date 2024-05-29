import streamlit as st
import pickle
import numpy as np
from datetime import datetime

startTime = datetime.now()

# Load the pre-trained model
filename = "model.h5"
with open(filename, 'rb') as file:
    model = pickle.load(file)

# Dictionaries for translating codes to labels
sex_d = {0: "Kobieta", 1: "Mężczyzna"}
pclass_d = {0: "Pierwsza", 1: "Druga", 2: "Trzecia"}
embarked_d = {0: "Cherbourg", 1: "Queenstown", 2: "Southampton"}

def main():
    st.set_page_config(page_title="Czy przeżyłbyś katastrofę?")
    overview = st.container()
    left, right = st.columns(2)
    prediction = st.container()

    with overview:
        st.title("Czy przeżyłbyś katastrofę?")
        st.image(
            "https://www.google.com/search?sca_esv=e87395d15064fe37&q=obesity&uds=ADvngMhpsojwKe5eIOqT5IDaiLbexbpZAAznK3Ol-4dTE8lWv2IWW5kf9ztqdsIKASuznHjntfzlXQ72wYgq6EgPDqKbQkTGVRD3GvDCn8RzmMBS3ExbH_iCc1PikDU-zM4ey0_V-YS6pdABTFHlS5ssOC7sdse-IGS_WDr12LrqrZ9DUPc2Bg_T7DggIHDYMSkVwj4FHjEvaeZYNUoZ05_mFOlz8H_xxew7k9cZrGvbxCSwZZlXhATusyReXWQk6iz5_NNcM-JeeYGjXdW2stIVCNQSTFNoyEsnJ-bxJhCDcDL5BMBjLfc&udm=2&prmd=ivnbz&sa=X&ved=2ahUKEwiq5YzeirOGAxWBJhAIHWpVCBIQtKgLegQIDhAB&biw=1872&bih=958&dpr=1#vhid=chjz49Y1fVQrmM&vssid=mosaic")

    with left:
        sex_radio = st.radio("Płeć", list(sex_d.keys()), format_func=lambda x: sex_d[x])
        pclass_radio = st.radio("Klasa", list(pclass_d.keys()), format_func=lambda x: pclass_d[x])
        embarked_radio = st.radio("Port zaokrętowania", list(embarked_d.keys()), index=2, format_func=lambda x: embarked_d[x])

    with right:
        age_slider = st.slider("Wiek", value=50, min_value=1, max_value=100)
        sibsp_slider = st.slider("# Liczba rodzeństwa i/lub partnera", min_value=0, max_value=8)
        parch_slider = st.slider("# Liczba rodziców i/lub dzieci", min_value=0, max_value=6)
        fare_slider = st.slider("Cena biletu", min_value=0, max_value=500, step=10)

    if st.button("Predict"):
        # Prepare input data for prediction
        data = np.array([[pclass_radio, age_slider, sibsp_slider, parch_slider, fare_slider, embarked_radio, sex_radio,]])

        # Predict survival
        survival = model.predict(data)
        s_confidence = model.predict_proba(data)

        with prediction:
            st.header(f"Czy dana osoba przeżyje? {'Tak' if survival[0] == 1 else 'Nie'}")
            st.subheader(f"Pewność predykcji {s_confidence[0][survival[0]] * 100:.2f} %")



if __name__ == "__main__":
    main()

## Źródło danych [https://www.kaggle.com/c/titanic/](https://www.kaggle.com/c/titanic), zastosowanie przez Adama Ramblinga
