import streamlit as st
import wikipediaapi
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

# Initialize the Wikipedia API with a user agent
wiki_wiki = wikipediaapi.Wikipedia('en', headers={'User-Agent': 'Tytanic/1.0 (s22078@pjwstk.edu.pl)'})

def fetch_wikipedia_page(title):
    page = wiki_wiki.page(title)
    if page.exists():
        return page.summary
    else:
        return "The page does not exist."

def main():
    st.set_page_config(page_title="Czy przeżyłbyś katastrofę?")

    # Sidebar for Wikipedia search
    st.sidebar.title("Wikipedia Search")
    search_term = st.sidebar.text_input("Enter a search term")

    if st.sidebar.button("Search"):
        if search_term:
            summary = fetch_wikipedia_page(search_term)
            st.sidebar.write(summary)
        else:
            st.sidebar.write("Please enter a search term.")

    overview = st.container()
    left, right = st.columns(2)
    prediction = st.container()

    with overview:
        st.title("Czy przeżyłbyś katastrofę?")

    with left:
        sex_radio = st.radio("Płeć", list(sex_d.keys()), format_func=lambda x: sex_d[x])
        pclass_radio = st.radio("Klasa", list(pclass_d.keys()), format_func=lambda x: pclass_d[x])
        embarked_radio = st.radio("Port zaokrętowania", list(embarked_d.keys()), index=2, format_func=lambda x: embarked_d[x])

    with right:
        age_input = st.text_input("Wiek", value="50")
        sibsp_input = st.text_input("# Liczba rodzeństwa i/lub partnera", value="0")
        parch_input = st.text_input("# Liczba rodziców i/lub dzieci", value="0")
        fare_input = st.text_input("Cena biletu", value="0")

    if st.button("Predict"):
        # Prepare input data for prediction
        data = np.array([[int(pclass_radio), int(sex_radio), int(age_input), int(sibsp_input), int(parch_input), float(fare_input), int(embarked_radio)]])

        # Predict survival
        survival = model.predict(data)
        s_confidence = model.predict_proba(data)

        with prediction:
            st.header(f"Czy dana osoba przeżyje? {'Tak' if survival[0] == 1 else 'Nie'}")
            st.subheader(f"Pewność predykcji {s_confidence[0][survival[0]] * 100:.2f} %")

    # Move the image to the bottom
    st.image("https://media1.popsugar-assets.com/files/thumbor/7CwCuGAKxTrQ4wPyOBpKjSsd1JI/fit-in/2048xorig/filters:format_auto-!!-:strip_icc-!!-/2017/04/19/743/n/41542884/5429b59c8e78fbc4_MCDTITA_FE014_H_1_.JPG")

if __name__ == "__main__":
    main()

## Źródło danych [https://www.kaggle.com/c/titanic/](https://www.kaggle.com/c/titanic), zastosowanie przez Adama Ramblinga
