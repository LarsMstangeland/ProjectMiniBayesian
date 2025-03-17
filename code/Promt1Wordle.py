
import pandas as pd
import numpy as np

# Load the CSV file into a DataFrame
file_Path = '../data/unigram_freq.csv'
df = pd.read_csv(file_Path)
count = 0


# trim to first 3000 rows
df = df.head(6)


def get_huffman_code(df = df.head(6)):

    # Dictionary to store the Huffman tree
    huffman_tree = {}


    #sorting count
    df = df.sort_values(by='count', ascending=False)


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



    print("\nFinal Huffman Tree Root Node:")
    print(df.head())
    print(f"Total Nodes in Final Tree: {len(df)}")



    # print(df.head())
    # print(len(df))

    # Print the Huffman tree dictionary
    print("\nHuffman Tree Structure:")
    for key, value in huffman_tree.items():
        print(f"{key}: {value}")

    # print how many nodes are in the tree



def Fano_code(df = df.head(6)):

    #sorting count
    df = df.sort_values(by='count', ascending=False)

    # Create a new column to store the Fano code
    df['fano_code'] = ""

    # Calculate the cumulative frequency
    df['cumulative_freq'] = df['count'].cumsum()

    # Calculate the total frequency
    total_freq = df['count'].sum()

    # Calculate the average frequency
    avg_freq = total_freq / len(df)

    

Fano_code()




















































