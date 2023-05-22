# Movie-Recomendation-System

A NLP system that recommends movies to the user using two techniques :
 -  Item based recommendation
 -  Content based recomendation

## Project Lifecycle

- Data Analysis
- Preprocessing 
- Modeling
- Deployment with streamlit

## Tech Stack 

Programming Languages: Python 3.9

Libraries used: NumPy, pandas, nltk, scentence_transformers, sklearn, matplotlib, streamlit, streamlit_lottie, seaborn, movieposters, requests, pickle.

## Data Analysis

![image](https://github.com/Nourhan-Mahmoud/Movie_Recommender_System-NLP_Project/assets/61950036/9e5fce26-6e83-44c8-9b78-0c05a3808183)

## Preprocessing 
### Content Based : 
 - Tokenization: The movie genres are tokenized by splitting them into individual words.
### Item Based :
 - pivoting : the ratings dataframe was pivoted to create a movie-user matrix where rows represent movies, columns represent users, and the values represent ratings.
 - Nulls : any null value in the ratings was filled by zero.

## Modeling
### Content Based : 
 - Word2Vec Training: The tokenized genres are used to train a Word2Vec model using Gensim. Word2Vec learns vector representations (embeddings) of words in a way that captures their semantic meaning based on the context in which they appear. The size of the embedding vectors is set to 100, and the minimum count is set to 1 to include all words.
 - Embedding: a given genre embedded by calculating the mean of the word embeddings for the individual words in the genre. If a word is not present in the Word2Vec model's vocabulary, a zero vector is used
 - Embedding Calculation: all the movie genres in the dataset embeddings were computed using the defined embedding function.
### Item Based :
 - K Nearest Neighbor : KNN was used with cosine similarity to find similar movies based on ratings in a function that predicts ratings for movies that a specific user has not watched. For each movie that the user has not rated, the function calculates a predicted rating by considering the ratings and similarities of similar movies. The predicted ratings are then filled in the copy of the original dataset.
 
## Prediction
### Content Based : 
 - Cosine Similarity: Cosine similarity is calculated between the genre embeddings to measure the similarity between different genres
 - Recommendation Function: A function is defined to provide movie recommendations based on a given movie index. It retrieves the cosine similarity scores for the specified movie and sorts them in descending order. The top 10 movies with the highest similarity scores are selected as recommendations. The function returns the recommended movies' titles and indices.
 - Output: The code demonstrates the recommendation function by providing recommendations for a specific movie index. It prints the watched movie's title, genres, and the top recommended movies' titles and genres.
### Item Based :
 - Movie Recommendations: the movies was recommendations were generated according to a selected user id. the ratings for unrated movies by the user were predicted. then the list of movies the user has already watched was retrieved and a list of recommended movies by sorting the unrated movies based on predicted ratings was generated ie top 10 recommended movies.

## Deployment
the system was deployed to a website using streamlit library. the website was designed for the examiners to choose one of the two methods ie. Item-Based and Content-Based. the examiner can search for the movie and the website would recommend the related movies after selecting the method.
