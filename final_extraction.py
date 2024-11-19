import csv
import ast
import numpy as np

def import_from_csv_precise(file_path):
    file = open(file_path,encoding='utf-8') # Put the name of the data file. Note: it shall be in the same folder as this file
    csvreader = csv.reader(file)
    header = next(csvreader)
    num_of_guesses=[]
    humor=[]
    levenshtein=[]
    glove=[]
    guesses=[]
    GPT_rating=[]
    for row in csvreader:
        guesses.append(ast.literal_eval(row[0]))
        num_of_guesses.append(ast.literal_eval(row[1]))
        humor.append(ast.literal_eval(row[2]))
        levenshtein.append(ast.literal_eval(row[3]))
        glove.append(ast.literal_eval(row[4]))
        GPT_rating.append(row[5])
    file.close()
    return guesses,num_of_guesses,humor,levenshtein,glove,GPT_rating

def characteristic_extraction(guesses,entropy,humor,levenshtein,glove,GPT_rating):
    # NEED TO ADD Y to input when getting that data
    final_embedding=[]
    guesses_valid=[]
    GPT_rating_valid=[]
    for i in range (len(entropy)):
        if (len(humor[i])==0 or len(levenshtein[i])==0 or len(glove[i])==0):
            continue
        if (len(entropy[i])<=1):
            continue
        embedding_for_game=[]
        # append in maximum, average and minimum of humor[i]
        embedding_for_game.append(max(humor[i]))
        embedding_for_game.append(sum(humor[i])/len(humor[i]))
        embedding_for_game.append(min(humor[i]))
        # append in maximum, average, minimum and last element of levenshtein[i]
        embedding_for_game.append(max(levenshtein[i]))
        embedding_for_game.append(sum(levenshtein[i])/len(levenshtein[i]))
        embedding_for_game.append(min(levenshtein[i]))
        embedding_for_game.append(levenshtein[i][-1])
        # append in maximum, average, minimum and last element of glove[i]
        embedding_for_game.append(max(glove[i]))
        embedding_for_game.append(sum(glove[i])/len(glove[i]))
        embedding_for_game.append(min(glove[i]))
        embedding_for_game.append(glove[i][-1])
        entropy_diff=[]
        for j in range(1,len(entropy[i])):
            entropy_diff.append(entropy[i][j-1]-entropy[i][j])
        # append in maximum, average, minimum and last element of entropy_diff
        embedding_for_game.append(max(entropy_diff))
        embedding_for_game.append(sum(entropy_diff)/len(entropy_diff))
        embedding_for_game.append(min(entropy_diff))
        embedding_for_game.append(entropy_diff[-1])
        final_embedding.append(embedding_for_game)
        guesses_valid.append(guesses[i])
        GPT_rating_valid.append(GPT_rating[i])
    return guesses_valid,final_embedding,GPT_rating_valid


guesses,entropy,humor,levenshtein,glove,GPT_rating=import_from_csv_precise('current_mapped_unique_wordle.csv')
guesses_valid,x_embedding,GPT_rating_valid=characteristic_extraction(guesses,entropy,humor,levenshtein,glove,GPT_rating)
#x_embedding=np.array(x_embedding)
GPT_rating_valid=np.array(GPT_rating_valid)
# do numpy save
np.save('x_embedding_unique.npy',x_embedding)
np.save('GPT_rating_unique.npy',GPT_rating_valid)
print(guesses_valid[1],x_embedding[1],GPT_rating_valid[1])
#print(guesses_valid[-1],x_embedding[-1],GPT_rating_valid[-1])
print(guesses_valid[5000],x_embedding[5000],GPT_rating_valid[5000])
#x_embedding=np.load('x_embedding_unique.npy')
#GPT_rating_valid=np.load('GPT_rating_valid.npy')
#print(x_embedding[5000],GPT_rating_valid[5000])
#print(x_embedding[1],GPT_rating_valid[1])