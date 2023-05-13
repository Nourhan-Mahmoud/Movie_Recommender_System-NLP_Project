import warnings
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import streamlit as st
import requests
import seaborn as sns
from streamlit_lottie import st_lottie
import pickle
from ItemBasedCF import recommend_movies_by_userId
import movieposters as mp

st.set_page_config(page_title='Movie Recommendation System',
                   page_icon=':star:', layout="wide")


def load_lottie(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


def retrieve_movie_name_genres(movie_id, movies_df):
    movie_name = movies_df[movies_df['movieId'] == movie_id]['title'].values[0]
    movie_genres = movies_df[movies_df['movieId']
                             == movie_id]['genres'].values[0]
    return movie_name, movie_genres


def get_watched_movies_info(watched_movies, movies_df, movie_cols):
    movies_titles = []
    movies_id = []
    movies_genres = []
    movies_poster_links = []
    for i in watched_movies[:10]:
        movie_name, movie_genres = retrieve_movie_name_genres(i, movies_df)
        movies_genres.append(movie_genres)
        movies_titles.append(movie_name)
        movies_id.append(i)
        try:
            link = mp.get_poster(title=movie_name)
            movies_poster_links.append(link)
        except:
            movies_poster_links.append(
                "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRoWcWg0E8pSjBNi0TtiZsqu8uD2PAr_K11DA&usqp=CAU")

    for c, i, t, p, g in zip(movie_cols, movies_id, movies_titles, movies_poster_links, movies_genres):
        with c:
            st.image(p)
            st.write(t)
            st.write("Movie ID: "+str(i))
            st.write("Genres: "+g)


def show_recommended_movie_info(recommended_movies, movie_cols, movies_df):
    movies_titles = []
    movies_id = []
    movies_genres = []
    movies_rank = []
    movies_rate = []
    movies_poster_links = []
    for i in recommended_movies:
        movie_name, movie_genres = retrieve_movie_name_genres(i[1], movies_df)
        movies_genres.append(movie_genres)
        movies_titles.append(movie_name)
        movies_rank.append(i[0])
        movies_rate.append(i[2])
        movies_id.append(i[1])
        try:
            link = mp.get_poster(title=movie_name)
            movies_poster_links.append(link)
        except:
            movies_poster_links.append(
                "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRoWcWg0E8pSjBNi0TtiZsqu8uD2PAr_K11DA&usqp=CAU")

    for c, i, t, r, p, rnk, g in zip(movie_cols, movies_id, movies_titles, movies_rate, movies_poster_links, movies_rank, movies_genres):
        with c:
            st.image(p)
            st.write(t)
            st.write("Movie ID: "+str(i))
            st.write("Genres: "+g)
            st.write("Predicted Rate: "+str(r))
            st.write("Rank: "+str(rnk))


st.markdown("<h1 style='text-align: center; color: black;'>Movie Recommender System</h1>",
            unsafe_allow_html=True)


tab1, tab2 = st.tabs(["Item-Based Collaborative Filtering",
                      "Content-Based Recommender System"])

with tab1:
    movieID_userID_df = pickle.load(open('moviesId_userId.pkl', 'rb'))
    users_ids = movieID_userID_df.columns.tolist()
    movies_df = pd.read_csv('../02-Code/movies.csv')
    col1, col2 = st.columns([1, 1])
    with col1:
        user = int(st.selectbox('Select a user', users_ids))
    with col2:
        num_movies = int(st.slider(
            'Select the number of movies to recommend', 1, 20, 10))

    if st.button('Show me the movies'):
        list_of_user_watched_movies, list_of_recommended_movies = recommend_movies_by_userId(
            movieID_userID_df, user, num_movies)

        with st.expander("Show me the movies I watched"):
            movie_cols = st.columns(len(list_of_user_watched_movies))
            get_watched_movies_info(
                list_of_user_watched_movies, movies_df, movie_cols)

        with st.expander("Top 10 Suggestions"):
            movie_cols = st.columns(num_movies)
            show_recommended_movie_info(
                list_of_recommended_movies, movie_cols, movies_df)

with tab2:
    st.markdown("<h3 style='text-align: center; color: black;'>Content-Based Recommender System</h3>",
                unsafe_allow_html=True)
