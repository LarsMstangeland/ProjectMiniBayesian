
import pandas as pd
import numpy as np
import math as Math

import os


# Load the CSV file into a DataFrame
file_Path = './data/unigram_freq.csv'
df = pd.read_csv(file_Path)
count = 0


# trim to first 3000 rows
df = df.sort_values(by='count', ascending=False)
df = df.head(31)

def get_huffman_code(df = df.head(6)):

    # Dictionary to store the Huffman tree
    huffman_tree = {}


    while len(df) > 1:

        #first group the two least frequent word
        word1 = df.iloc[-1]
        word2 = df.iloc[-2]

        #Get cumulative frequency of the two least frequent words
        sum_freq = word1['count'] + word2['count']
        
        #Make a set of two least frequent words
        LeastWordsSet = np.array([word1['word'], word2['word']])
        LeastWordsSetWithFreqent = np.append(LeastWordsSet, sum_freq)

        #Create Key
        # Key1 = df.index[df['word'] == word1['word']].tolist()
        # Key2 = df.index[df['word'] == word2['word']].tolist()
        # UniqueKey = str(Key1[0]) +'_'+ str(Key2[0])
        UniqueKey = f"{word1['word']}_{word2['word']}"

        # Store parent-child relationship in Huffman tree
        huffman_tree[UniqueKey] = {"left": word1['word'], "right": word2['word'], "frequency": sum_freq}

   
        #dictionary of the words
        Set1 = {UniqueKey: sum_freq}

        # Create a new DataFrame for the new row
        new_row = pd.DataFrame({'word': [UniqueKey], 'count': [sum_freq]})
        
        #remove the last two words
        df = df.drop(df.tail(2).index)

        df = pd.concat([df, new_row], ignore_index=True)


        # Sort the DataFrame by count
        df = df.sort_values(by='count', ascending=False)

        #print(f"Merged {word1['word']} and {word2['word']} -> {UniqueKey} ({sum_freq})")



    # print("\nFinal Huffman Tree Root Node:")
    # print(df.head())
    # print(f"Total Nodes in Final Tree: {len(df)}")



    # print(df.head())
    # print(len(df))

    #Print the Huffman tree dictionary
    # print("\nHuffman Tree Structure:")
    # for key, value in huffman_tree.items():
    #     print(f"{key}: {value}")

    # print how many nodes are in the tree

    return huffman_tree

def make_huffman_code(huffman_tree, prefix, node, huffman_code):
    if node not in huffman_tree:
        huffman_code[node] = prefix
        return

    left = huffman_tree[node]['left']
    right = huffman_tree[node]['right']

    make_huffman_code(huffman_tree, prefix + "0", left, huffman_code)
    make_huffman_code(huffman_tree, prefix + "1", right, huffman_code)

    return huffman_code


# tree = get_huffman_code()
# root = list(tree.keys())[-1]
# print(make_huffman_code(tree, "", root, {}))







def Fano_prosedure(df = df.head(6)):
    """
    Finds the best split for Fano code
    """
    #sorting count

    # Create a new column to store the Fano code
    df['fano_code'] = ""

    # Calculate the cumulative frequency
    df['cumulative_freq'] = df['count'].cumsum()

    # Calculate the total frequency
    total_freq = df['count'].sum()


    # Calculate the average frequency
    df['Normalized_cf'] = df['cumulative_freq'] / total_freq

    min_index = (df['Normalized_cf']-0.5).abs().idxmin()
    # print(df)
    # print(f"min index:{min_index}")
    return min_index
    

fano_tree={}

def Fano_code(df, prefix):



   
    df = df.sort_values(by='count', ascending=False)


    #Split the dataframe
    smallest_index = Fano_prosedure(df)

    
    if len(df) > 1:
        left = df.iloc[:smallest_index+1].reset_index(drop=True)
        right = df.iloc[smallest_index+1:].reset_index(drop=True)
        

    else:
        word = df.iloc[0]['word']
        fano_tree[word] = prefix
        return 

    
    Fano_code(left, prefix + "0")
    Fano_code(right, prefix + "1")


    


    
    return fano_tree






def get_probability_for_word(word, df = df):
    """
    Get the probability of a word in the DataFrame
    """
    word_row = df[df['word'] == word]
    if len(word_row) == 0:
        return 0
    return word_row['count'].values[0] / df['count'].sum()


word_probs = {word: get_probability_for_word(word) for word in df['word']}

# print(word_probs)

def get_steps_in_fano(word, fano_tree):
    """
    Get the number of steps in Fano code for a word
    """
    return len(fano_tree[word])



def average_code_length_fano(fano_tree, word_probs):
    """
    Calculate the average code length for Fano code
    """
    return sum([get_steps_in_fano(word, fano_tree) * word_probs[word] for word in word_probs])

print(f"Average code length for Fano code: {average_code_length_fano(Fano_code(df, prefix=''), word_probs)}")


def get_steps_in_huffman(word, huffman_tree):
    """
    Get the number of steps in Huffman code for a word
    """
    return len(huffman_tree[word])

def average_code_length_huffman(huffman_tree, word_probs):
    """
    Calculate the average code length for Huffman code
    """
    return sum([get_steps_in_huffman(word, huffman_tree) * word_probs[word] for word in word_probs])

huffman_tree = get_huffman_code()
root_node = list(huffman_tree.keys())[-1]
huffman_code = make_huffman_code(huffman_tree, "", root_node, {})
print(f"Average code length for Huffman code: {average_code_length_huffman(huffman_code, word_probs)}")












# print(f"Average code length for Huffman code: {average_code_length_huffman(make_huffman_code(get_huffman_code()), word_probs)}")














