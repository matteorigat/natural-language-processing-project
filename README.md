This project presents a novel system for generating culinary recipes based on
user-provided ingredients. Leveraging the power of pre-trained Large Language
Models (LLMs) like GPT-2 and statistical analysis techniques, the system intel-
ligently combines ingredients with suitable cooking methods to generate a recipe.
The system is trained on a large dataset of recipes, learning culinary contexts and
correlations between ingredients and cooking methods. This knowledge enables
the generation of novel recipes that adhere to culinary norms and are likely to
be well-received. To generate a recipe the system first use a trained word2vec
model to insert in the generation also the cooking methods suitable for the pro-
vided ingredients. Generated recipes are finally evaluated against a benchmark of
similar real recipes and their user ratings, ensuring the system’s output is both
creative and of high quality.
