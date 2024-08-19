import openai
from dotenv import dotenv_values
import pandas as pd
import numpy as np
from tenacity import retry, wait_random_exponential, stop_after_attempt
import pickle
import tiktoken
from scipy import spatial
from typing import List
import argparse

# Load environment variables from the .env file
config = dotenv_values(".env")
openai.api_key = config["OPENAI_API_KEY"]

# Load the movie dataset
dataset_path = "./movie_plots.csv"
df = pd.read_csv(dataset_path)

# Narrow down the dataset to 5000 recent American movies to optimize cost
movies = (
    df[df["Origin/Ethnicity"] == "American"]
    .sort_values("Release Year", ascending=False)
    .head(5000)
)

# Extract the movie plots and titles into separate lists
movie_plots = movies["Plot"].values
movie_titles = movies["Title"].values

# Define a function to get embeddings from OpenAI API with retry logic
@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def get_embedding(text, model="text-embedding-ada-002"):
    # Replace newlines, which can negatively affect performance
    text = text.replace("\n", " ")
    
    # Generate and return the embedding
    return openai.embeddings.create(input=text, model=model).data[0].embedding

# Estimate the cost of generating embeddings for all movie plots
enc = tiktoken.encoding_for_model("text-embedding-ada-002")
total_tokens = sum([len(enc.encode(plot)) for plot in movie_plots])
cost = total_tokens * (0.0004 / 1000)
print(f"Estimated cost ${cost:.2f}")

# Establish a cache of embeddings to avoid recomputing
embedding_cache_path = "movie_embeddings_cache2.pkl"

# Load the cache if it exists, or create an empty cache
try:
    embedding_cache = pd.read_pickle(embedding_cache_path)
except FileNotFoundError:
    embedding_cache = {}

# Save a copy of the cache to disk
with open(embedding_cache_path, "wb") as embedding_cache_file:
    pickle.dump(embedding_cache, embedding_cache_file)

# Define a function to retrieve embeddings from the cache or request via the API
def embedding_from_string(
    string, model="text-embedding-ada-002", embedding_cache=embedding_cache
):
    """Return embedding of given string, using a cache to avoid recomputing."""
    if (string, model) not in embedding_cache.keys():
        embedding_cache[(string, model)] = get_embedding(string, model)
        print(f"GOT EMBEDDING FROM OPENAI FOR {string[:20]}")
        with open(embedding_cache_path, "wb") as embedding_cache_file:
            pickle.dump(embedding_cache, embedding_cache_file)
    return embedding_cache[(string, model)]

# Generate the embeddings for all movie plots
plot_embeddings = [
    embedding_from_string(plot, model="text-embedding-ada-002") for plot in movie_plots
]

# Visualize the embeddings using the Atlas tool
data = movies[["Title", "Genre"]].to_dict("records")
from nomic import atlas
project = atlas.map_embeddings(embeddings=np.array(plot_embeddings), data=data)

# Define a function to calculate distances between embeddings
def distances_from_embeddings(
    query_embedding: List[float],
    embeddings: List[List[float]],
    distance_metric="cosine",
) -> List[List]:
    """Return the distances between a query embedding and a list of embeddings."""
    distance_metrics = {
        "cosine": spatial.distance.cosine,
        "L1": spatial.distance.cityblock,
        "L2": spatial.distance.euclidean,
        "Linf": spatial.distance.chebyshev,
    }
    distances = [
        distance_metrics[distance_metric](query_embedding, embedding)
        for embedding in embeddings
    ]
    return distances

# Define a function to get indices of nearest neighbors from distances
def indices_of_nearest_neighbors_from_distances(distances) -> np.ndarray:
    """Return a list of indices of nearest neighbors from a list of distances."""
    return np.argsort(distances)

# Define a function to print movie recommendations based on a given title
def print_recommendations_from_title(
    plots,
    titles,
    input_title,
    k_nearest_neighbors=3,
    model="text-embedding-ada-002",
):
    # Find the index of the input title
    index_of_source_string = list(titles).index(input_title)
    
    # Get all of the embeddings
    embeddings = [embedding_from_string(plot) for plot in plots]
    
    # Get embedding for the specific query string
    query_embedding = embeddings[index_of_source_string]
    
    # Get distances between the query embedding and all other embeddings
    distances = distances_from_embeddings(query_embedding, embeddings)
    
    # Get indices of the nearest neighbors
    indices_of_nearest_neighbors = indices_of_nearest_neighbors_from_distances(distances)

    # Print the recommendations
    query_string = plots[index_of_source_string]
    match_count = 0
    for i in indices_of_nearest_neighbors:
        if query_string == plots[i]:
            continue
        if match_count >= k_nearest_neighbors:
            break
        match_count += 1
        print(f"Found {match_count} closest match: ")
        print(f"Distance of: {distances[i]:.4f} ")
        print(f"Title: {titles[i]}")
        print(f"Plot: {plots[i]}")

# Main function to parse command-line arguments and call the recommendation function
def main():
    parser = argparse.ArgumentParser(description="Movie Recommendations based on Plot")
    parser.add_argument("-p", "--title", type=str, required=True, help="Title of the movie")
    parser.add_argument("-n", "--number", type=int, default=3, help="Number of recommendations to generate")
    
    args = parser.parse_args()
    
    print_recommendations_from_title(movie_plots, movie_titles, args.title, args.number)

if __name__ == "__main__":
    main()
