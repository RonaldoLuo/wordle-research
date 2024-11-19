import gensim.downloader as api
from numpy.linalg import norm
import numpy as np
import csv
import ast

# Load the pre-trained GloVe model (300 dimensions, trained on Wikipedia + Gigaword)
model = api.load('glove-wiki-gigaword-300')

# Function to calculate cosine similarity between two vectors
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))

# Function to compute distance between two words using GloVe
def glove_distance(word1, word2, model):
    if word1 in model and word2 in model:
        vec1 = model[word1]
        vec2 = model[word2]
        
        # Calculate Cosine Similarity
        similarity = cosine_similarity(vec1, vec2)
        distance = 1 - similarity  # Cosine distance
        
        return similarity, distance
    else:
        return f"One or both words ('{word1}', '{word2}') not found in the GloVe model."

def levenshtein_between_guesses(source,target):
    if (len(source)==0):
        return len(target)
    if (len(target)==0):
        return len(source)
    if (source[0]==target[0]):
        return levenshtein_between_guesses(source[1:],target[1:])
    direct_edit=levenshtein_between_guesses(source[1:],target[1:])
    insert=levenshtein_between_guesses(source,target[1:]) # insert the same alphabit as the start of target
    delete=levenshtein_between_guesses(source[1:],target) # delete the starting alphabit of source
    return 1+ min(delete,min(direct_edit,insert))

def import_from_csv(file_path):
    file = open(file_path,encoding='utf-8') # Put the name of the data file. Note: it shall be in the same folder as this file
    csvreader = csv.reader(file)
    header = next(csvreader)
    X=[]
    y=[]
    data=[]
    for row in csvreader:
        X.append(row[0])
        # change data from string to float
        y.append(float(row[-1]))
        data.append(row)
    file.close()
    return X,y,data

def import_from_csv_guesses(file_path):
    file = open(file_path,encoding='utf-8') # Put the name of the data file. Note: it shall be in the same folder as this file
    csvreader = csv.reader(file)
    header = next(csvreader)
    data=[]
    for row in csvreader:
        data.append(ast.literal_eval(row[2]))
    file.close()
    return data

def import_from_csv_data(file_path):
    file = open(file_path,encoding='utf-8') # Put the name of the data file. Note: it shall be in the same folder as this file
    csvreader = csv.reader(file)
    header = next(csvreader)
    data=[]
    for row in csvreader:
        data.append(row)
    file.close()
    return data

def characteristic_extraction(guesses,word_list,humor_score,model):
    humor_ratings=[]
    levenshtein_datas=[]
    glove_distance_datas=[]
    count=0
    for i in range (len(guesses)):
        game=guesses[i]
        count+=1
        humor_score_for_game=[]
        levenshtein_data_for_game=[]
        glove_distance_data_for_game=[]
        for guess in game:
            if (guess in word_list):
                humor_score_for_game.append(humor_score[word_list.index(guess)])
        for j in range(1,len(game)):
            levenshtein_data_for_game.append(levenshtein_between_guesses(game[j-1],game[j]))
            if (game[j-1] in model and game[j] in model):
                glove_distance_data_for_game.append(glove_distance(game[j-1],game[j],model)[1])
        humor_ratings.append(humor_score_for_game)
        levenshtein_datas.append(levenshtein_data_for_game)
        glove_distance_datas.append(glove_distance_data_for_game)
        if (count%100==0):
            print(count)
    return humor_ratings,levenshtein_datas,glove_distance_datas

def write_to_csv(file_path,data,humor_score,levenshtein_data,glove_distance_data):
    file = open(file_path, 'w',encoding='utf-8',newline='')
    # write data into file
    writer = csv.writer(file)
    writer.writerow(('comment','distance_from_parent','wordle_words','num_possible_guesses','intrinsic_humor_of_words','levenshtein_distance','glove_distance'))
    for i in range(len(data)):
        writer.writerow((data[i][0],data[i][1],data[i][2],data[i][3],humor_score[i],levenshtein_data[i],glove_distance_data[i]))

guesses=import_from_csv_guesses('wordle_funny_structure_with_entropy.csv')
original_data=import_from_csv_data('wordle_funny_structure_with_entropy.csv')
word_list,humor_score,_=import_from_csv('rating_new.csv')
humor_d,leven_d,glove_d=characteristic_extraction(guesses,word_list,humor_score,model)
write_to_csv('wordle_final.csv',original_data,humor_d,leven_d,glove_d)
