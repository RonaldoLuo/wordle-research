from sklearn.linear_model import Ridge
import csv
import math
import gensim.downloader as api
from numpy.linalg import norm
import numpy as np
import copy
import ast
import math
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
# Function to calculate cosine similarity between two vectors
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))

# Function to compute distance between two words using Word2Vec
def word2vec_distance(word1, word2, model):
    if word1 in model and word2 in model:
        vec1 = model[word1]
        vec2 = model[word2]
        
        # Calculate Cosine Similarity
        similarity = cosine_similarity(vec1, vec2)
        distance = 1 - similarity  # Cosine distance
        
        return similarity, distance
    else:
        return f"One or both words ('{word1}', '{word2}') not found in the Word2Vec model."

def compute_semantic_benchmark_for_six_categroy(model,X):
    # Define the six categories
    categories = {
        'party': ['banter', 'joke', 'giggle', 'chuckle','laughter','antics','chitchat','shindig','dinner','ditty','soiree','bash','frolic','lark','booze','supper','buddy','brunch','toast'],
        'sex': ['pubes', 'boob', 'pussy', 'crotch','pecker','cunt','dildo','penis','panties','twat','knickers','hussy','butt','fuck','douche','cleavage','fucker','babe','puss'],
        'insult': ['moron','idiot','cretin','buffoon','jackass','twat','bastard','weirdo','dude','douche','slob','fucker','bitch','chump','whore','ninny','blockhead','cunt','stupid'],
        'profanity': ['shit','fuck','moron','damn','twat','fucker','bitch','bullshit','dude','bastard','douche','idiot','piss','cunt','stupid','jackass','puke','turd','bollocks'],
        'body function': ['puke','puss','pussy','pubes','butt','crotch','slobber','pecker','cunt','snot','vomit','twat','penis','shit','turd','fucker','anus','douche','scrotum'],
        'animals':['critter','puppy','reptile','feline','pooch','raccoon','bobcat','lizard','tortoise','possum','collie','terrier','monkey','piglet','chimp','otter','giraffe','hippo','cheetah']        
    }
    categories_benchmark_final=[]
    # iterate through dictionary categories
    for key in categories:
        semantic_info=model[categories[key]]
        # compute average semantic embedding using model
        column_wise_average = np.mean(semantic_info, axis=0)
        similarity_array=[]
        for word in X:
            similarity_array.append(cosine_similarity(model[word],column_wise_average))
        sorted_indices = np.argsort(similarity_array,axis=None)[::-1]
        portion=np.array(sorted_indices[:100])
        new_X=np.array(X)
        new_description_vectors=model[new_X[portion]]
        final_benmark=np.mean(new_description_vectors,axis=0)
        categories_benchmark_final.append(final_benmark)
        # sort array similarity_array in decreasing order and get index
    # save categories_benchmark_final
    np.save('benchmark.npy', categories_benchmark_final)

def test(word,model):
    # load the benchmark from the numpy file
    categories_benchmark_final = np.load('benchmark.npy')
    # compare word with each benchmark
    word_embedding=model[word]
    similarity_array=[]
    for benchmark in categories_benchmark_final:
        similarity_array.append(cosine_similarity(word_embedding,benchmark))
    return similarity_array

def compute_benchmark_for_le_ending(model):
    target=['gaggle','jiggle','tinkle','wiggle','waddle','wriggle','gobble','nibble']
    semantic_info=model[target]
    # compute average semantic embedding using model
    column_wise_average = np.mean(semantic_info, axis=0)
    np.save('le_benchmark.npy', column_wise_average)
    return column_wise_average

def import_from_csv(file_path):
    file = open(file_path) # Put the name of the data file. Note: it shall be in the same folder as this file
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

def import_from_csv_col2(file_path):
    file = open(file_path, encoding="utf8") # Put the name of the data file. Note: it shall be in the same folder as this file
    csvreader = csv.reader(file)
    header = next(csvreader)
    y=[]
    for row in csvreader:
        # get the second element in the row and change that element from string to array
        y.append(ast.literal_eval(row[1]))
    file.close()
    return y

def import_from_csv_utf8(file_path):
    file = open(file_path, encoding='utf-8') # Put the name of the data file. Note: it shall be in the same folder as this file
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

def import_from_txt(file_path):
    file = open(file_path, encoding='utf-8') # Put the name of the data file. Note: it shall be in the same folder as this file
    data = file.read().splitlines()
    X=[]
    y=[]
    for row in data:
        row=row.split(",")
        X.append(row[0])
        y.append(row[-1])
    file.close()
    return X,y

def create_txt_file(file_path,data):
    file = open(file_path, 'w') # Put the name of the data file. Note: it shall be in the same folder as this file
    for row in data:
        file.write(row)
        file.write("\n")
    file.close()

def check_k_in_word(word):
    return 'k' in word

def check_u(word,words,pronounce):
    index=words.index(word)
    return 'u' in pronounce[index]
    
def get_valence_arousal_domiance_and_concreteness(word,word_list,characteristics):
    index=word_list.index(word)
    return characteristics[index]

# uses a different data set than the one used in paper as this one is newer and has more data
# although will make the frequnecy value higher, but shall not have any effect as the weights will be adjusted accordingly
def check_freq_of_word(word,dict_words,dict_freq):
    # find index of word in the np array dict_words
    dict_words=np.array(dict_words)
    index = np.nonzero(dict_words==word)[0]
    index=index[0]
    return math.log(dict_freq[index])

def check_character_frequency_of_word(word,dict_letter,letter_freq):
    sum=0
    length=len(word)
    dict_letter=np.array(dict_letter)
    for char in word:
        index = np.nonzero(dict_letter==char)[0][0]
        sum+=letter_freq[index]        
    return math.log(sum/100/length)

def check_phoneme_frequency_of_word(word,dict_phon,phon_freq,words,pronounce):
    sum=0
    length=0
    words=np.array(words)
    print(word)
    words_index = np.nonzero(words==word)[0][0]
    pronounce=pronounce[words_index]
    pronounce_new=copy.deepcopy(pronounce)
    dict_phon=np.array(dict_phon)
    # check if any element in dict_phon is present in pronounce_new
    for phon in dict_phon:
        while(phon in pronounce_new):
            index = np.nonzero(dict_phon==phon)[0][0]
            sum+=phon_freq[index]
            length+=1
            # remove phon from pronounce_new
            pronounce_new=pronounce_new.replace(phon,"",1)
    return math.log(sum/100/length)

def import_from_csv_with_multiple_ys(file_path):
    file = open(file_path) # Put the name of the data file. Note: it shall be in the same folder as this file
    csvreader = csv.reader(file)
    header = next(csvreader)
    X=[]
    y=[]
    data=[]
    for row in csvreader:
        X.append(row[0])
        # change data from string to float
        y.append(row[1:])
        data.append(row)
    file.close()
    return X,y,data

def not_in_norm(word,word_list):
    return word not in word_list

def not_in_phon(word,words):
    return word not in words

def list_not_in_target(words,word_list,word_list2):
    new_words=[]
    both_not_in=[]
    phono_not_in=[]
    for word in words:
        if word not in word_list:
            new_words.append(word)
        if word not in word_list2:
            phono_not_in.append(word)
        if word not in word_list and word not in word_list2:
            both_not_in.append(word)
    return np.unique(new_words),np.unique(phono_not_in),np.unique(both_not_in)

def number_of_words_not_in_target(words,dict_words,wordlist2,model,wordlist):
    # count number of words not each in seperately and combined using for loop
    count_tol=0
    count_model=0
    count_phono=0
    count_norm=0
    count_dict=0
    for word in words:
        if word not in dict_words or word not in wordlist2 or word not in wordlist or word not in model:
            count_tol+=1
        if word not in dict_words:
            count_dict+=1
        if word not in model:
            count_model+=1
        if word not in wordlist2:
            count_phono+=1
        if word not in wordlist:
            count_norm+=1
    return count_tol,count_model,count_phono,count_norm,count_dict

def data_preprocessing(X,y,dict_words,dict_frequency,dict_letter,letter_freq,dict_phon,phon_freq,words,pronounce,word_list,charcteristics,model):
    processed_X=[]
    new_y=[]
    categories_benchmark_final = np.load('benchmark.npy')
    le_benchmark_final = np.load('le_benchmark.npy')
    for i in range(len(X)):
        word=X[i]
        if (word not in model or word not in dict_words or word not in words or word not in word_list): 
            continue
        description=[]
        word_embedding=model[word]
        for benchmark in categories_benchmark_final:
            description.extend([1-cosine_similarity(word_embedding,benchmark)])
        description.extend([int(check_k_in_word(word.lower()))])
        description.extend([check_freq_of_word(word.lower(),dict_words,dict_frequency)])
        description.extend([check_character_frequency_of_word(word.lower(),dict_letter,letter_freq)])
        description.extend([check_phoneme_frequency_of_word(word.lower(),dict_phon,phon_freq,words,pronounce)])
        description.extend([int(check_u(word.lower(),words,pronounce))])
        description.extend([1-cosine_similarity(word_embedding,le_benchmark_final)])
        description.extend(get_valence_arousal_domiance_and_concreteness(word.lower(),word_list,charcteristics))
        processed_X.append(description)
        new_y.append(y[i])
    # save processed_X as a numpy file
    np.save('processed_X.npy', processed_X)
    # save new_y as a numpy file
    np.save('new_y.npy', new_y)
    return processed_X,new_y

def data_preprocessing_alternative(X,y,dict_words,dict_frequency,dict_letter,letter_freq,dict_phon,phon_freq,words,pronounce,word_list,charcteristics,model):
    processed_X=[]
    new_y=[]
    categories_benchmark_final = np.load('benchmark.npy')
    le_benchmark_final = np.load('le_benchmark.npy')
    for i in range(len(X)):
        word=X[i]
        if (word not in model or word not in dict_words or word not in words or word not in word_list): 
            continue
        description=[]
        word_embedding=model[word]
        sum=0
        for benchmark in categories_benchmark_final:
            description.extend([1-cosine_similarity(word_embedding,benchmark)])
        description.extend([int(check_k_in_word(word.lower()))])
        description.extend([check_freq_of_word(word.lower(),dict_words,dict_frequency)])
        description.extend([check_character_frequency_of_word(word.lower(),dict_letter,letter_freq)])
        description.extend([check_phoneme_frequency_of_word(word.lower(),dict_phon,phon_freq,words,pronounce)])
        description.extend([check_character_frequency_of_word(word.lower(),dict_letter,letter_freq)
                            /check_phoneme_frequency_of_word(word.lower(),dict_phon,phon_freq,words,pronounce)])
        description.extend([int(check_u(word.lower(),words,pronounce))])
        description.extend([1-cosine_similarity(word_embedding,le_benchmark_final)])
        description.extend([ast.literal_eval(get_valence_arousal_domiance_and_concreteness(word.lower(),word_list,charcteristics)[0])
                            *ast.literal_eval(get_valence_arousal_domiance_and_concreteness(word.lower(),word_list,charcteristics)[1])])
        description.extend([ast.literal_eval(get_valence_arousal_domiance_and_concreteness(word.lower(),word_list,charcteristics)[0])
                            *ast.literal_eval(get_valence_arousal_domiance_and_concreteness(word.lower(),word_list,charcteristics)[1])
                            *ast.literal_eval(get_valence_arousal_domiance_and_concreteness(word.lower(),word_list,charcteristics)[2])])
        description.extend([ast.literal_eval(get_valence_arousal_domiance_and_concreteness(word.lower(),word_list,charcteristics)[1])
                            *ast.literal_eval(get_valence_arousal_domiance_and_concreteness(word.lower(),word_list,charcteristics)[2])])
        description.extend([get_valence_arousal_domiance_and_concreteness(word.lower(),word_list,charcteristics)[0]])  
        description.extend([get_valence_arousal_domiance_and_concreteness(word.lower(),word_list,charcteristics)[1]])        
        description.extend([get_valence_arousal_domiance_and_concreteness(word.lower(),word_list,charcteristics)[3]])
        processed_X.append(description)
        new_y.append(y[i])
    # save processed_X as a numpy file
    np.save('processed_X2.npy', processed_X)
    # save new_y as a numpy file
    np.save('new_y2.npy', new_y)
    return processed_X,new_y

def train_linear_regression(X, Y):
    # initiate an object of the LinearRegression type. 
    reg=Ridge(alpha=1, fit_intercept=True)
    r_square=0
    train_error=0
    test_error=0
    # do cross validation
    for i in range(10):
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
        # run the fit function to train the model. 
        reg=reg.fit(X_train,y_train) 
        # run the predict function to get the predictions
        y_pred=reg.predict(X_train)
        train_error+=math.sqrt(np.mean((y_pred - y_train) ** 2))        
        w = reg.coef_
        intercept=reg.intercept_
        np.save('weights.npy', w)
        np.save('bias.npy', intercept)
        test_error+=test(X_test,y_test)
        r_square+=reg.score(X,Y)
        # saving weights trained by the model.

    print("R^2: ",r_square/10)
    print("Train Error: ",train_error/10)
    print("Test Error: ",test_error/10)
    return r_square/10,train_error/10,test_error/10

def train_linear_regression_alternative(X, Y):
    # initiate an object of the LinearRegression type. 
    reg=Ridge(alpha=1, fit_intercept=True)
    r_square=0
    train_error=0
    test_error=0
    # do cross validation
    for i in range(10):
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
        # run the fit function to train the model. 
        reg=reg.fit(X_train,y_train) 
        # run the predict function to get the predictions
        y_pred=reg.predict(X_train)
        train_error+=math.sqrt(np.mean((y_pred - y_train) ** 2))        
        w = reg.coef_
        intercept=reg.intercept_
        np.save('weights2.npy', w)
        np.save('bias2.npy', intercept)
        test_error+=test_alternative(X_test,y_test)
        r_square+=reg.score(X,Y)
        # saving weights trained by the model.

    print("R^2: ",r_square/10)
    print("Train Error: ",train_error/10)
    print("Test Error: ",test_error/10)
    return r_square/10,train_error/10,test_error/10

def predict_data(X_in):
    # load the weights and bias from the numpy file
    w = np.load('weights.npy')
    b=np.load('bias.npy')
    # add 1 to the beginning of each row in X_in
    y_pred = np.dot(X_in, w)+b
    return y_pred

def test(X_test, y_test):
    # load the weights and bias from the numpy file
    w = np.load('weights.npy')
    b=np.load('bias.npy')
    # add 1 to the beginning of each row in X_test
    y_pred = np.dot(X_test, w)+b
    # calculate the mean square loss
    return math.sqrt(np.mean((y_pred - y_test) ** 2))

def test_alternative(X_test, y_test):
    # load the weights and bias from the numpy file
    w = np.load('weights2.npy')
    b=np.load('bias2.npy')
    # add 1 to the beginning of each row in X_test
    y_pred = np.dot(X_test, w)+b
    # calculate the mean square loss
    return math.sqrt(np.mean((y_pred - y_test) ** 2))

def histogram_of_funny_words_rating(y,y_pred):
    bins=np.arange(0,100,5)
    plt.hist([y,y_pred], bins=bins,label=['Actual','Predicted'],color=['blue','red'])
    plt.legend(loc='upper right')
    # name the x-axis
    plt.xlabel("Value")
    # name the y-axis
    plt.ylabel("Frequency")
    # create title
    plt.title("Histogram of Funny Words Rating")
    plt.show()

def scatter_plot_of_funny_words_rating(y,y_pred):
    plt.rcParams.update({'font.size': 12})
    plt.scatter(y,y_pred)
    # create fitted linear line
    plt.plot(np.unique(y), np.poly1d(np.polyfit(y, y_pred, 1))(np.unique(y)), color='red')
    print(np.poly1d(np.polyfit(y, y_pred, 1)))
    # add the equation of fitted linear line on the graph 
    plt.text(75, 52.5, 'y = 0.3707x + 34.95', color='black')
    # name the x-axis
    plt.xlabel("Actual")
    # make x-axis label bigger
    # name the y-axis
    plt.ylabel("Predicted")
    # create title
    plt.title("Scatter Plot of Funny Words Rating")
    # make text bigger

    plt.show()

def input_data_preprocessing(target,dict_words,dict_frequency,dict_letter,letter_freq,dict_phon,phon_freq,words,pronounce,word_list,charcteristics,model):
    processed_X=[]
    availuable_words=[]
    categories_benchmark_final = np.load('benchmark.npy')
    le_benchmark_final = np.load('le_benchmark.npy')
    for i in range(len(target)):
        word=target[i]
        if (word not in model or word not in dict_words or word not in words or word not in word_list): 
            continue
        description=[]
        word_embedding=model[word]
        for benchmark in categories_benchmark_final:
            description.extend([1-cosine_similarity(word_embedding,benchmark)])
        description.extend([int(check_k_in_word(word.lower()))])
        description.extend([check_freq_of_word(word.lower(),dict_words,dict_frequency)])
        description.extend([check_character_frequency_of_word(word.lower(),dict_letter,letter_freq)])
        description.extend([check_phoneme_frequency_of_word(word.lower(),dict_phon,phon_freq,words,pronounce)])
        description.extend([check_character_frequency_of_word(word.lower(),dict_letter,letter_freq)
                            /check_phoneme_frequency_of_word(word.lower(),dict_phon,phon_freq,words,pronounce)])
        description.extend([int(check_u(word.lower(),words,pronounce))])
        description.extend([1-cosine_similarity(word_embedding,le_benchmark_final)])
        description.extend([ast.literal_eval(get_valence_arousal_domiance_and_concreteness(word.lower(),word_list,charcteristics)[0])
                            *ast.literal_eval(get_valence_arousal_domiance_and_concreteness(word.lower(),word_list,charcteristics)[1])])
        description.extend([ast.literal_eval(get_valence_arousal_domiance_and_concreteness(word.lower(),word_list,charcteristics)[0])
                            *ast.literal_eval(get_valence_arousal_domiance_and_concreteness(word.lower(),word_list,charcteristics)[1])
                            *ast.literal_eval(get_valence_arousal_domiance_and_concreteness(word.lower(),word_list,charcteristics)[2])])
        description.extend([ast.literal_eval(get_valence_arousal_domiance_and_concreteness(word.lower(),word_list,charcteristics)[1])
                            *ast.literal_eval(get_valence_arousal_domiance_and_concreteness(word.lower(),word_list,charcteristics)[2])])
        description.extend([ast.literal_eval(get_valence_arousal_domiance_and_concreteness(word.lower(),word_list,charcteristics)[0])]) 
        description.extend([ast.literal_eval(get_valence_arousal_domiance_and_concreteness(word.lower(),word_list,charcteristics)[1])])        
        description.extend([ast.literal_eval(get_valence_arousal_domiance_and_concreteness(word.lower(),word_list,charcteristics)[3])])
        processed_X.append(description)
        availuable_words.append(word)
    # save processed_X as a numpy file
    np.save('embedding.npy', processed_X)
    # save new_y as a numpy file
    np.save('availuable_words.npy', availuable_words)
    return availuable_words,processed_X

def predict_input(X_in):
    # load the weights and bias from the numpy file
    w = np.load('weights2.npy')
    b=np.load('bias2.npy')
    # add 1 to the beginning of each row in X_in
    y_pred = np.dot(X_in, w)+b
    return y_pred


def create_csv(file_path,word,rating):
    # create a csv with header words and rating 
    with open(file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["words", "rating"])
        for i in range(len(word)):
            writer.writerow([word[i], rating[i]])

#dict_words,dict_frequency,_=import_from_csv("unigram_freq.csv")
#X,y,_=import_from_csv("humor.csv")
#dict_letter,letter_freq,_=import_from_csv("english_monograms.csv")
#make dict_letter lowercase
##dict_letter=[x.lower() for x in dict_letter]
#word_list,descriptions,_=import_from_csv_with_multiple_ys("Hollisetal_norms.csv")
# make word_list lowercase
#word_list=[x.lower() for x in word_list]
#print(check_k_in_word("kick"))
#print(check_freq_of_word("the",dict_words,dict_frequency))
#print(check_character_frequency_of_word("kick",dict_letter,letter_freq))
# Load pre-trained Word2Vec model from Google News
#model = api.load('word2vec-google-news-300')
#compute_semantic_benchmark_for_six_categroy(model,X)
#print(test("lizard",model))
#print(word2vec_distance("lizard", "puppy", model))
#a=compute_benchmark_for_le_ending(model)
#print(cosine_similarity(model["gaggle"],a))
#dict_phon,phon_freq,_=import_from_csv_utf8("phonemes.csv")
#words,pronounce=import_from_txt("CMU.in.IPA.txt")
#print(check_phoneme_frequency_of_word("kier",dict_phon,phon_freq,words,pronounce))
#print(check_u("poof",words,pronounce))
#print(X)
# processed_X,new_y=data_preprocessing(X,y,dict_words,dict_frequency,dict_letter,letter_freq,dict_phon,phon_freq,words,pronounce,word_list,descriptions,model)
#processed_X_2,new_y_2=data_preprocessing_alternative(X,y,dict_words,dict_frequency,dict_letter,letter_freq,dict_phon,phon_freq,words,pronounce,word_list,descriptions,model)
#print(processed_X_2[0],new_y_2[0])
#print(np.array(processed_X).shape)
# split data into train and test
# load X train
#processed_X = np.load('processed_X2.npy')
#new_processed_X=np.zeros((len(processed_X),len(processed_X[0])))
# convert processed_X to numeric values
#for i in range(len(processed_X)):
#   for j in range(len(processed_X[0])):
#        new_processed_X[i][j]=ast.literal_eval(processed_X[i][j])

# load y train
#new_y = np.load('new_y2.npy')
#print(new_processed_X.shape)
#print(new_y.shape)
#print(new_y.min(),new_y.max())
# train and test the model
#train_linear_regression_alternative(new_processed_X,new_y)
#y_pred=predict_input(new_processed_X)
#histogram_of_funny_words_rating(new_y,y_pred)
#scatter_plot_of_funny_words_rating(new_y,y_pred)

#target=import_from_csv_col2("wordle_funny_structure_new.csv")
#print(target[0],target[1])
# concatenate word
#target=np.concatenate(target)
#target=np.unique(target)
#target=[x.lower() for x in target]

#not_in_norm_words,phono,both=list_not_in_target(target,word_list,words)
# create txt file for not_in_norm_words
#create_txt_file("infile.txt",not_in_norm_words)
# print len of target
#print(len(target))
# print lens of each
#print(len(not_in_norm_words),len(phono),len(both))
# print number of words not in target
#count_tol,count_model,count_phono,count_norm,dict_count=number_of_words_not_in_target(target,dict_words,words,model,word_list)
#print(count_tol,count_model,count_phono,count_norm,dict_count)

# pre-process target
#availuable_words,embedding=input_data_preprocessing(target,dict_words,dict_frequency,dict_letter,letter_freq,dict_phon,phon_freq,words,pronounce,word_list,descriptions,model)
#availuable_words=np.load('availuable_words.npy')
#embedding=np.load('embedding.npy')
#print(availuable_words[3500],embedding[3500])
# predict input
#y_pred=predict_input(embedding)
# create csv file
#create_csv("rating_new.csv",availuable_words,y_pred)

# test whether rating_new.csv and rating.csv hold same item
#rating_new_words,rating,_=import_from_csv("rating_new.csv")
#rating_words,rating_old,_=import_from_csv("rating.csv")
#print(rating_words==rating_new_words)
#print(rating==rating_old)
