import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors


def predict_rating_for_movies_that_user_not_watched(df, user_id, number_neighbors):
    # copy df
    df1 = df.copy()

    # find the nearest neighbors using NearestNeighbors
    number_neighbors = 10
    knn = NearestNeighbors(metric='cosine', algorithm='brute')
    knn.fit(df.values)
    distances, indices = knn.kneighbors(
        df.values, n_neighbors=number_neighbors)

    # convert user_name to user_index
    user_index = df.columns.tolist().index(user_id)

    #  m: the row number of t in df,t: movie_title
    for m, t in list(enumerate(df.index)):

        # find movies without ratings by user_id
        if df.iloc[m, user_index] == 0:
            similar_movies = indices[m].tolist()
            movie_distances = distances[m].tolist()

            # indices[3] = [3 6 7]. The movie itself is in the first place. so we remove the movie itself from the list.
            if m in similar_movies:
                id_movie = similar_movies.index(m)
                similar_movies.remove(m)
                movie_distances.pop(id_movie)

            else:
                similar_movies = similar_movies[:number_neighbors-1]
                movie_distances = movie_distances[:number_neighbors-1]

            # movie_similarity = 1 - movie_distance
            movie_similarity = [1-x for x in movie_distances]
            movie_similarity_copy = movie_similarity.copy()
            nominator = 0

            for s in range(0, len(movie_similarity)):
                # check if the rating of a similar movie is zero
                if df.iloc[similar_movies[s], user_index] == 0:

                    # if the rating is zero, ignore the rating and the similarity in calculating the predicted rating
                    if len(movie_similarity_copy) == (number_neighbors - 1):
                        movie_similarity_copy.pop(s)
                    else:
                        movie_similarity_copy.pop(
                            s-(len(movie_similarity)-len(movie_similarity_copy)))

                # if the rating is not zero, use the rating and similarity in the calculation
                else:
                    nominator = nominator + \
                        movie_similarity[s] * \
                        df.iloc[similar_movies[s], user_index]

            # check if the number of the ratings with non-zero is positive
            if len(movie_similarity_copy) > 0:
                # check if the sum of the ratings of the similar movies is positive.
                if sum(movie_similarity_copy) > 0:
                    predicted_r = nominator/sum(movie_similarity_copy)
                # Even if there are some movies for which the ratings are positive, some movies have zero similarity even though they are selected as similar movies.
                # in this case, the predicted rating becomes zero as well
                else:
                    predicted_r = 0
            # if all the ratings of the similar movies are zero, then predicted rating should be zero
            else:
                predicted_r = 0

            # place the predicted rating into the copy of the original dataset
            df1.iloc[m, user_index] = round(predicted_r, 1)
    return df1


def recommend_movies_by_userId(df, user, num_recommended_movies):
    df1 = predict_rating_for_movies_that_user_not_watched(df, user, 1000)

    list_of_user_watched_movies = df[df[user] > 0][user].index.tolist()

    recommended_movies = []

    for m in df[df[user] == 0].index.tolist():
        index_df = df.index.tolist().index(m)
        predicted_rating = df1.iloc[index_df, df1.columns.tolist().index(user)]
        recommended_movies.append((m, predicted_rating))

    sorted_recommended_movies = sorted(
        recommended_movies, key=lambda x: x[1], reverse=True)

    list_of_recommended_movies = []
    rank = 1
    for recommended_movie in sorted_recommended_movies[:num_recommended_movies]:
        list_of_recommended_movies.append(
            [rank, recommended_movie[0], recommended_movie[1]])
        rank = rank + 1

    return list_of_user_watched_movies, list_of_recommended_movies


def give_recommendations_with_contentBased(index, cos_sim_data, data, num_movies=10):
    index_recomm = cos_sim_data.loc[index].sort_values(
        ascending=False).index.tolist()[0:num_movies]
    index = [int(i) for i in index_recomm]
    movies_recomm = data['title'].loc[index].values
    movies_recomm_genres = data['genres'].loc[index].values
    return movies_recomm, index_recomm, movies_recomm_genres
