import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 1. Import libraries and create rating data
ratings = pd.DataFrame({
    "Movie A": [5, 4, 0, 0, 2],
    "Movie B": [0, 5, 3, 2, 0],
    "Movie C": [0, 0, 5, 4, 4],
    "Movie D": [2, 2, 4, 5, 4],
    "Movie E": [3, 0, 0, 0, 5]
}, index=["User 1", "User 2", "User 3", "User 4", "User 5"])

print("Movie Rating Table:")
print(ratings)

# 2. User-based Collaborative Filtering
user_similarity = cosine_similarity(ratings)
user_similarity = pd.DataFrame(
    user_similarity,
    index=ratings.index,
    columns=ratings.index
)
print("User Similarity:")
print(user_similarity.round(2))

user = "User 1"
# Find similar users
similar_users = user_similarity[user].drop(user)
# Calculate recommendation scores
other_users_ratings = ratings.drop(index=user)
scores = other_users_ratings.T.dot(similar_users)
# Select movies not rated by User 1
unrated = ratings.loc[user] == 0
scores = scores[unrated]
# Display top 2 recommendations
print("Top 2 User-based Recommendations:")
print(scores.sort_values(ascending=False).head(2))

# 3. Item-based Collaborative Filtering
item_similarity = cosine_similarity(ratings.T)
item_similarity = pd.DataFrame(
    item_similarity,
    index=ratings.columns,
    columns=ratings.columns
)
print("Movie Similarity:")
print(item_similarity.round(2))

# Ratings given by User 1
user_ratings = ratings.loc["User 1"]
# Calculate scores for each movie
scores = item_similarity.dot(user_ratings)
# Do not recommend movies already rated
unrated = user_ratings == 0
scores = scores[unrated]
# Display top 2 recommendations
print("Top 2 Item-based Recommendations:")
print(scores.sort_values(ascending=False).head(2))
