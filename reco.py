import argparse
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def extract_names(x): 
    try:
        return ' '.join([d['name'] for d in eval(x)])
    except:
        return ''

def load_data(csv_path):
    """Load and preprocess movie dataset with multiple features"""
    df = pd.read_csv(csv_path)
    
    # Handle missing values
    df['overview'] = df['overview'].fillna('')
    df['tagline'] = df['tagline'].fillna('')
    
    for col in ['genres', 'keywords', 'production_companies', 'production_countries']:
        df[col] = df[col].apply(extract_names)
    
    # Create combined text feature
    df['metadata'] = (
        df['overview'] + ' ' +
        df['tagline'] + ' ' +
        df['genres'] + ' ' +
        df['keywords'] + ' ' +
        df['original_language'] + ' ' +
        df['production_companies']
    )
    
    return df[['title', 'metadata']].dropna().reset_index(drop=True)

def build_tfidf_matrix(df):
    """Create TF-IDF vectors with enhanced preprocessing"""
    tfidf = TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 2),
        max_features=1000
    )
    tfidf_matrix = tfidf.fit_transform(df['metadata'])
    return tfidf_matrix, tfidf

def recommend_movies(query, df, tfidf_matrix, vectorizer, n=5):
    """Generate movie recommendations based on text similarity"""
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = similarities.argsort()[::-1][:n]
    
    return [
        (df.iloc[i]['title'], similarities[i])
        for i in top_indices
        if similarities[i] > 0
    ]

def main():
    parser = argparse.ArgumentParser(description="Movie Recommendation System")
    parser.add_argument('query', type=str, help='Search query for movie recommendations')
    
    args = parser.parse_args()
    data_path = 'tmdb_5000_movies.csv'
    movies_df = load_data(data_path)
    tfidf_matrix, vectorizer = build_tfidf_matrix(movies_df)
    recommendations = recommend_movies(args.query, movies_df, tfidf_matrix, vectorizer, 5)

    # Fixed print statements
    print("\nTop 5 recommendations for '{}'".format(args.query))
    for title, score in recommendations:
        print("- {} (score: {:.3f})".format(title, score))

if __name__ == "__main__":
    main()