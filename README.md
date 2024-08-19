# 🎥 Movie Recommendation System Using OpenAI Embeddings
This project demonstrates a movie recommendation system using OpenAI's text embeddings model (text-embedding-ada-002). The system generates movie recommendations based on plot similarity by embedding movie plots into high-dimensional vector space. This project showcases the use of embeddings for semantic search and recommendation tasks.

## 📊 Dataset
The dataset used for this project is sourced from [Kaggle's Wikipedia Movie Plots dataset](https://www.kaggle.com/datasets/jrobischon/wikipedia-movie-plots). The dataset contains details of over 34,000 movies, including their plots. For this project, we narrowed down the dataset to 5,000 American movies from recent years to focus on and reduce computational costs.

## 🌟 Features
- Embedding Generation: Convert movie plots into 1536-dimensional embeddings using OpenAI's text-embedding-ada-002.
- Movie Recommendation: Find movies with similar plots using cosine similarity.
- Data Visualization: Visualize the movie embeddings in 2D using the [Atlas tool](https://www.kaggle.com/datasets/jrobischon/wikipedia-movie-plots), showcasing the natural clustering of similar movies.

## 🚀 Getting Started
### Prerequisites
To run this project, you'll need:

- Python 3.7 or higher
- OpenAI API key
- Kaggle API key to download the dataset
- Required Python libraries: openai, dotenv, pandas, numpy, tenacity, tiktoken, scipy, nomic, atlas

### Installation
1. Clone the repository:
```
git clone https://github.com/yourusername/movierecommendation.git
cd movierecommendation
```

2. Install the required libraries:
```
pip install -r requirements.txt
```

3. Set up your .env file:
Create a .env file in the root directory and add your OpenAI API key:
```
OPENAI_API_KEY=your_openai_api_key
```

4. Download the dataset:
Download the dataset from Kaggle:
[Wikipedia Movie Plots Dataset](https://www.kaggle.com/datasets/jrobischon/wikipedia-movie-plots)
Place the movie_plots.csv file in the root directory.

### Usage
1. Generate Embeddings:
Run the script to generate embeddings for the movie plots:
```
python app.py
```

2. Get Movie Recommendations:
Use the following command to get movie recommendations based on a specific movie title:
```
python3 app.py -p "Title of the Movie" -n 5
```
Replace "Title of the Movie" with the movie of your choice. This will display 5 similar movie recommendations based on plot similarity.

3. Visualize the Embeddings:
You can explore the movie plot embeddings in 2D using the [Atlas tool](https://www.kaggle.com/datasets/jrobischon/wikipedia-movie-plots). This interactive map shows how the movies are clustered based on plot similarities.


## 📈 Data Visualization
Check out the Atlas tool to explore the embedding space. Each point represents a movie, and similar movies naturally form clusters.
![Screenshot of movie clusters.](/atlas.png)

## 🤝 Contributing
If you'd like to contribute to this project, feel free to submit pull requests or open issues. Your contributions are welcome!

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
