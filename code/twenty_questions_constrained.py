import pandas as pd
import numpy as np
import gensim.downloader as api


#######################################################################################################################
##### a)
#######################################################################################################################

# Load the data from the CSV file
file_path = './data/unigram_freq.csv'
df = pd.read_csv(file_path, keep_default_na=False)

# Ttrim to the top 3000 most frequent words
df = df.sort_values(by='count', ascending=False).head(3000)

# Load the GloVe word embeddings model
model = api.load('glove-wiki-gigaword-50')

# dictionary to store the normalized embeddings for the words
word_embeddings = {}
missing_words = []

# For each word in the DataFrame, we will attempt to get its embedding
for word in df['word']:
    if word in model:
        embedding = model[word]
        # Normalize the embedding to have unit norm (i.e. project onto the unit sphere)
        norm = np.linalg.norm(embedding)
        if norm != 0:
            word_embeddings[word] = embedding / norm
        else:
            missing_words.append(word)
    else:
        missing_words.append(word)

print(f"Embeddings found for {len(word_embeddings)} out of {len(df)} words.")
print(f"Example missing words: {missing_words[:10]}")  # printing some of the missing words

# we create a DataFrame that contains only the words with available embeddings
df_embedded = df[df['word'].isin(word_embeddings.keys())].reset_index(drop=True)
print("DataFrame of words with embeddings:")
print(df_embedded.head())

# the word_embeddings dictionary has the normalized embeddings for the words, with df_embedded being the corresponding DataFrame.


#######################################################################################################################
##### b)
#######################################################################################################################


# --------------------------------------------------------------------
# 1. Project words onto the unit sphere and compute priors
# --------------------------------------------------------------------
# df_embedded have the top 3,000 words with embeddings.
# word_embeddings is a dictionary mapping each word to its normalized embedding.
total_count = df_embedded['count'].sum()
df_embedded = df_embedded.copy()  # create a local copy to avoid modifying the original
df_embedded['prob'] = df_embedded['count'] / total_count  # weight words by their frequency (prior probability)
word_probs = dict(zip(df_embedded['word'], df_embedded['prob']))

# this function is used to sample a random unit vector (used for candidate hyperplanes)
def sample_random_unit_vector(dim):
    v = np.random.randn(dim)
    return v / np.linalg.norm(v)

# --------------------------------------------------------------------
# 2. Iteratively identify a hyperplane that splits the current probability mass aprpox. 50/50
# --------------------------------------------------------------------
def find_best_hyperplane(candidates, word_embeddings, word_probs, num_candidates=100):
    dim = len(next(iter(word_embeddings.values())))
    best_n = None
    best_diff = float('inf')
    for _ in range(num_candidates):
        # this samples a random unit vector
        n = sample_random_unit_vector(dim)
        # here we compute the probability mass on the "positive" side (i.e. where dot product >= 0)
        pos_mass = sum(word_probs[w] for w in candidates if np.dot(word_embeddings[w], n) >= 0)
        diff = abs(pos_mass - 0.5)  # we want a split as close as possible to 50/50
        if diff < best_diff:
            best_diff = diff
            best_n = n
    return best_n

# --------------------------------------------------------------------
# 3, 4, and 5. Ask the question, drop wrong-side words, and iterate until resolution
# --------------------------------------------------------------------
# def simulate_constrained_question_procedure(df_embedded, word_embeddings, word_probs):
#     # Create the list of candidate words from our DataFrame (in 1. we already handled the projection and priors)
#     candidates = list(df_embedded['word'])
    
#     # we then sample an unknown word from the candidates using the prior probabilities
#     words = np.array(candidates)
#     probs = np.array([word_probs[w] for w in candidates])
#     unknown = np.random.choice(words, p=probs)
#     print("Unknown word (drawn from prior):", unknown)
    
#     steps = 0
#     # Set a variable to track if we are not reducing the candidate set
#     stagnation_count = 0
#     # Define a threshold for how many iterations with no reduction we'll tolerate
#     STAGNATION_THRESHOLD = 5  # you can adjust this as needed
#     # we continue the process until we have only one candidate left
#     while len(candidates) > 1:
#         steps += 1
        
#         # we find a hyperplane that best splits the probability mass among the current candidates
#         n = find_best_hyperplane(candidates, word_embeddings, word_probs, num_candidates=100)
        
#         # we check if the unknown word is on the same side as the hyperplane
#         unknown_side = np.dot(word_embeddings[unknown], n) >= 0
        
#         # we update the candidate list based on the side of the hyperplane
#         if unknown_side:
#             new_candidates = [w for w in candidates if np.dot(word_embeddings[w], n) >= 0]
#         else:
#             new_candidates = [w for w in candidates if np.dot(word_embeddings[w], n) < 0]
        
#         # Debug output to show how the probability mass is split.
#         pos_mass = sum(word_probs[w] for w in candidates if np.dot(word_embeddings[w], n) >= 0)
#         neg_mass = 1 - pos_mass
#         print(f"Step {steps}: {len(candidates)} candidates; split: {pos_mass:.3f} vs. {neg_mass:.3f}.")


#         # we check if the candidate set has shrunk significantly.
#         if len(new_candidates) < len(candidates):
#             stagnation_count = 0  # Reset if we made progress
#         else:
#             stagnation_count += 1  # No reduction detected
        
#         # Fallback: if we have several iterations without reduction, choose the highest-probability candidate
#         if stagnation_count >= STAGNATION_THRESHOLD:
#             print("Candidate set did not shrink after several iterations. Using fallback selection.")
#             chosen_candidate = max(candidates, key=lambda w: word_probs[w])
#             print("Fallback candidate chosen:", chosen_candidate)
#             return chosen_candidate
        
#         # Updatate of the candidate list with the words that remain after dropping the wrong side.
#         candidates = new_candidates
#         print(f"After step {steps}, {len(candidates)} candidates remain.\n")
    
#     print("Final candidate:", candidates[0])
#     print("Total questions asked:", steps)


#a run of the procedure
# simulate_constrained_question_procedure(df_embedded, word_embeddings, word_probs)

## comments: in some runs of the simulation, we get a heavy-tailed distributino of the word ferquencies.
## therefore, we added a check in the iterative loop, such that if after an iteration the candidate set hasn’t reduced in size, we break the loop and choose the candidate with the highest prior probability.
## The fallback mechanism is a way to handle cases where the hyperplane splits are not effective in reducing the candidate set.



# Modified simulation function that returns the number of steps (questions) taken
def simulate_constrained_question_procedure_return_steps(df_embedded, word_embeddings, word_probs, stagnation_threshold=5, num_candidates=100):
    # Start with all candidate words
    candidates = list(df_embedded['word'])
    # Sample an unknown word using the prior probabilities
    words = np.array(candidates)
    probs = np.array([word_probs[w] for w in candidates])
    unknown = np.random.choice(words, p=probs)
    
    steps = 0
    stagnation_count = 0  # tracks consecutive iterations with no reduction in candidate count
    
    while len(candidates) > 1:
        steps += 1
        n = find_best_hyperplane(candidates, word_embeddings, word_probs, num_candidates=num_candidates)
        # Determine the side of the hyperplane where the unknown word lies
        unknown_side = np.dot(word_embeddings[unknown], n) >= 0
        # Partition candidates based on the hyperplane's side
        if unknown_side:
            new_candidates = [w for w in candidates if np.dot(word_embeddings[w], n) >= 0]
        else:
            new_candidates = [w for w in candidates if np.dot(word_embeddings[w], n) < 0]
        
        # Check if the candidate set is reduced
        if len(new_candidates) < len(candidates):
            stagnation_count = 0
        else:
            stagnation_count += 1
        
        # Fallback: if no reduction for several iterations, select the highest-probability candidate
        if stagnation_count >= stagnation_threshold:
            chosen_candidate = max(candidates, key=lambda w: word_probs[w])
            candidates = [chosen_candidate]
            break
        
        candidates = new_candidates
        
    return steps
    


#######################################################################################################################
##### c)
#######################################################################################################################

# Run the simulation multiple times to estimate the expected number of questions
num_trials = 100  # we adjust this number as needed
steps_list = []
for i in range(num_trials):
    steps = simulate_constrained_question_procedure_return_steps(df_embedded, word_embeddings, word_probs)
    steps_list.append(steps)

avg_steps = np.mean(steps_list)
std_steps = np.std(steps_list)
print(f"Average number of questions over {num_trials} trials: {avg_steps:.2f} ± {std_steps:.2f}")

# Compute Shannon entropy of the prior distribution as a lower bound
def shannon_entropy(word_probs):
    H = 0
    for p in word_probs.values():
        if p > 0:
            H -= p * np.log2(p)
    return H

H = shannon_entropy(word_probs)
print(f"Shannon entropy (lower bound on questions): {H:.2f}")


