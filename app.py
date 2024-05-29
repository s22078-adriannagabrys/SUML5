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
        return page.fullurl
    else:
        return None

def main():
    st.set_page_config(page_title="Czy przeżyłbyś katastrofę?")

    # Sidebar for Wikipedia search
    st.sidebar.title("Wikipedia Search")
    search_term = st.sidebar.text_input("Enter a search term")

    if st.sidebar.button("Search"):
        if search_term:
            wikipedia_url = fetch_wikipedia_page(search_term)
            if wikipedia_url:
                st.sidebar.write(f"Displaying Wikipedia page for '{search_term}'")
                st.sidebar.components.v1.html(f'<iframe src="{wikipedia_url}" width="100%" height="600" style="border:none;"></iframe>', scrolling=True)
            else:
                st.sidebar.write("Page not found on Wikipedia.")
        else:
            st.sidebar.write("Please enter a search term.")

    overview = st.container()
    left, right = st.columns(2)
    prediction = st.container()

    with overview:
        st.title("Czy przeżyłbyś katastrofę?")

    with left:
        sex_radio = st.radio("Płeć", {0: "Kobieta", 1: "Mężczyzna"})
        pclass_radio = st.radio("Klasa", {0: "Pierwsza", 1: "Druga", 2: "Trzecia"})
        embarked_radio = st.radio("Port zaokrętowania", {0: "Cherbourg", 1: "Queenstown", 2: "Southampton"})

    with right:
        age_slider = st.slider("Wiek", value=50, min_value=1, max_value=100)
        sibsp_slider = st.slider("# Liczba rodzeństwa i/lub partnera", min_value=0, max_value=8)
        parch_slider = st.slider("# Liczba rodziców i/lub dzieci", min_value=0, max_value=6)
        fare_slider = st.slider("Cena biletu", min_value=0, max_value=500, step=10)

    if st.button("Predict"):
        # Prepare input data for prediction
        data = np.array([[pclass_radio, sex_radio, age_slider, sibsp_slider, parch_slider, fare_slider, embarked_radio]])

        # Load the pre-trained model
        filename = "model.h5"
        with open(filename, 'rb') as file:
            model = pickle.load(file)

        # Predict survival
        survival = model.predict(data)
        s_confidence = model.predict_proba(data)

        with prediction:
            st.header(f"Czy dana osoba przeżyje? {'Tak' if survival[0] == 1 else 'Nie'}")
            st.subheader(f"Pewność predykcji {s_confidence[0][survival[0]] * 100:.2f} %")
