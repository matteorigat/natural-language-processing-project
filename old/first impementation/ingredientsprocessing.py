import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import re
from gensim.models import Word2Vec
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import nltk

# Download NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# Define the list of cooking techniques
TECHNIQUES_LIST = [
    'bake', 'boil', 'braise', 'broil', 'fry', 'grill', 'roast', 'saute', 'simmer',
    'steam', 'stew'
]

# Define a new technique mapping based on TECHNIQUES_LIST
technique_mapping = {
    'bake': 'bake',
    'barbecue': 'grill',
    'blanch': 'boil',
    'blend': 'combine',
    'boil': 'boil',
    'braise': 'braise',
    'brine': 'marinate',
    'broil': 'broil',
    'caramelize': 'saute',
    'combine': 'combine',
    'crock pot': 'slow cook',
    'crush': 'puree',
    'deglaze': 'saute',
    'devein': 'prepare',
    'dice': 'prepare',
    'distill': 'boil',
    'drain': 'boil',
    'emulsify': 'mix',
    'ferment': 'bake',
    'freeze': 'refrigerate',
    'fry': 'fry',
    'grate': 'saute',
    'griddle': 'prepare',
    'grill': 'grill',
    'knead': 'prepare',
    'leaven': 'bake',
    'marinate': 'marinate',
    'mash': 'stew',
    'melt': 'melt',
    'microwave': 'stew',
    'parboil': 'bake',
    'pickle': 'pickling',
    'poach': 'stew',
    'pour': 'combine',
    'pressure cook': 'bake',
    'puree': 'puree',
    'refrigerate': 'refrigerate',
    'roast': 'roast',
    'saute': 'saute',
    'scald': 'fry',
    'scramble': 'combine',
    'shred': 'puree',
    'simmer': 'simmer',
    'skillet': 'saute',
    'slow cook': 'slow cook',
    'smoke': 'grill',
    'smooth': 'stew',
    'soak': 'bake',
    'sous-vide': 'combine',
    'steam': 'steam',
    'stew': 'stew',
    'strain': 'bake',
    'tenderize': 'fry',
    'thicken': 'boil',
    'toast': 'grill',
    'toss': 'stew',
    'whip': 'grill',
    'whisk': 'stew'
}


# Load the model and other necessary data
def load_model_and_data(model_path, data_path):
    # Load the trained Word2Vec model
    model = Word2Vec.load(model_path)

    # Load data and other necessary components
    data = pd.read_csv(data_path)

    # Process data to get ingredient counts and mappings
    raw_ingredients = []
    raw_techniques = []

    for index, row in data.iterrows():
        ingredients = eval(row['ingredients'])
        raw_ingredients.extend(ingredients)

        techniques = eval(row['techniques_list'])
        raw_techniques.extend(techniques)

    ingredient_counts = Counter(raw_ingredients)

    technique_corpus = [[technique] for technique in raw_techniques]
    tokenized_techniques = [technique.split() for technique in raw_techniques]

    technique_model = Word2Vec(tokenized_techniques, vector_size=100, min_count=1, workers=4, window=5)

    technique_vectors = {technique: np.mean([technique_model.wv[word] for word in technique.split()], axis=0) for
                         technique in raw_techniques}

    scaler = MinMaxScaler()
    scaled_technique_vectors = scaler.fit_transform(list(technique_vectors.values()))

    num_clusters_techniques = len(TECHNIQUES_LIST)
    kmeans_technique_clustering = KMeans(n_clusters=num_clusters_techniques)
    technique_idx = kmeans_technique_clustering.fit_predict(scaled_technique_vectors)

    technique_cluster_map = {technique: cluster for technique, cluster in zip(technique_vectors.keys(), technique_idx)}

    dominant_techniques_per_cluster = {cluster: Counter(
        [technique for technique, cluster_id in technique_cluster_map.items() if cluster_id == cluster]).most_common(1)[
        0][0] for cluster in range(num_clusters_techniques)}
    cluster_to_method = {cluster: technique for cluster, technique in dominant_techniques_per_cluster.items()}

    ingredient_cluster_map = {ingredient: kmeans_technique_clustering.predict(
        scaler.transform([np.mean([model.wv[word] for word in ingredient.split()], axis=0)]))[0] for ingredient in
                              set(raw_ingredients)}

    return model, scaler, ingredient_counts, technique_cluster_map, cluster_to_method, ingredient_cluster_map


# Define the cleaning function
def clean_ingredient(ingredient, regex, stop_words, lemmatizer, common_ingredients_to_avoid):
    cleaned_ingredient = regex.sub('', ingredient.lower()).strip()
    cleaned_words = []

    for word in cleaned_ingredient.split():
        if word in common_ingredients_to_avoid:
            cleaned_words = []
            break
        elif len(word) > 2 and word not in stop_words:
            cleaned_words.append(lemmatizer.lemmatize(word))

    return cleaned_words


# Define the function to get dominant cooking methods
def get_dominant_cooking_methods(ingredients, ingredient_cluster_map, cluster_to_method, ingredient_counts, model,
                                 scaler):
    method_counts = Counter()
    total_ingredient_count = sum(ingredient_counts.values())

    # Clean and filter ingredients
    cleaned_ingredients = []
    for ingredient in ingredients:
        cleaned_words = clean_ingredient(ingredient, re.compile('[^a-zA-Z ]'), set(stopwords.words('english')),
                                         WordNetLemmatizer(),
                                         ['salt', 'sugar', 'water', 'pepper', 'butter', 'onion', 'garlic', 'oil',
                                          'parsley'])
        if cleaned_words:  # Check if cleaned_words is not empty
            cleaned_ingredients.append(" ".join(cleaned_words))

    # Calculate relative importance weights for each ingredient based on its frequency
    weights = {ingredient: ingredient_counts.get(ingredient, 0) / total_ingredient_count for ingredient in
               cleaned_ingredients}

    # Calculate method counts using the mapped techniques
    for ingredient in cleaned_ingredients:
        cluster = ingredient_cluster_map.get(ingredient)
        if cluster is not None:
            method = cluster_to_method.get(cluster)
            if method:
                mapped_method = technique_mapping.get(method)
                if mapped_method:
                    method_counts[mapped_method] += weights[ingredient]

    # Normalize method counts
    if method_counts:
        max_count = max(method_counts.values())
        method_counts = {method: count / max_count for method, count in method_counts.items()}

    # Get top 3 methods from TECHNIQUES_LIST
    top_methods = sorted(method_counts.items(), key=lambda x: x[1], reverse=True)[:3]

    # Determine the most important ingredient based on weights
    most_important_ingredient = max(cleaned_ingredients, key=lambda x: weights.get(x, 0))

    return [method for method, _ in top_methods], most_important_ingredient


# Define the main function to process a CSV and generate a new CSV
def process_ingredients_csv(input_csv_path, output_csv_path, model_path, data_path):
    # Load the model and data
    model, scaler, ingredient_counts, technique_cluster_map, cluster_to_method, ingredient_cluster_map = load_model_and_data(
        model_path, data_path)

    # Read the input CSV
    df = pd.read_csv(input_csv_path)

    # Initialize the list to store recommended methods
    recommended_methods = []

    # Process each row in the CSV
    for index, row in df.iterrows():
        ingredients = row['ingredients'].split(',')
        methods, _ = get_dominant_cooking_methods(ingredients, ingredient_cluster_map, cluster_to_method,
                                                  ingredient_counts, model, scaler)
        recommended_methods.append(", ".join(methods))

    # Add the recommended methods to the DataFrame
    df['recommended_methods'] = recommended_methods

    # Save the new DataFrame to a CSV
    df.to_csv(output_csv_path, index=False)

# Example usage:
# process_ingredients_csv('input_ingredients.csv', 'output_recommended_methods.csv', 'word2vec_model_path', 'data_path')
