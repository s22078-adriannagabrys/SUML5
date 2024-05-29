import streamlit as st
import pickle
import numpy as np
from datetime import datetime
import wikipediaapi

# Initialize the Wikipedia API
wiki_wiki = wikipediaapi.Wikipedia('en')

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
        age_input = st.text_input("Wiek", value="50")
        sibsp_input = st.text_input("# Liczba rodzeństwa i/lub partnera", value="0")
        parch_input = st.text_input("# Liczba rodziców i/lub dzieci", value="0")
        fare_input = st.text_input("Cena biletu", value="0")

    if st.button("Predict"):
        # Prepare input data for prediction
        data = np.array([[int(pclass_radio), int(sex_radio), int(age_input), int(sibsp_input), int(parch_input), float(fare_input), int(embarked_radio)]])

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

if __name__ == "__main__":
    main()

## Źródło danych [https://www.kaggle.com/c/titanic/](https://www.kaggle.com/c/titanic), zastosowanie przez Adama Ramblinga
