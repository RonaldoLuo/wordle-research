from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
import ast
import csv
from sklearn.neighbors import KNeighborsClassifier

def custom_predict(X, threshold,model):
    probs = model.predict_proba(X) 
    return (probs[:, 1] > threshold).astype(int)
    
    
def train_logistic_regression(X, Y):
    # initiate an object of the LinearRegression type. 
    regression=LogisticRegression(penalty=None,fit_intercept=True,max_iter=50000)
    # do cross validation
    train_accuracy=0
    test_accuracy=0
    accuracy_choosing_dominant_class=0
    for i in range(10):
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
        newdata = X_train
        new_y_train = y_train
        # run the fit function to train the model. 
        # split the data by making sure 1s and 0s of Y are split in 50/50
        # datapoint=sum(y_train=='1')
        # newdata=np.vstack((X_train[y_train=='0'][0:datapoint],X_train[y_train=='1']))
        # newdata=np.array(newdata)
        # new_y_train=np.hstack((y_train[y_train=='0'][0:datapoint],y_train[y_train=='1']))
        # new_y_train=np.array(new_y_train)
        regression=regression.fit(newdata,new_y_train) 
        # run the predict function to get the predictions
        y_pred=custom_predict(newdata,0.265,regression)

        # convert binary to string predictions
        y_pred=[str(y) for y in y_pred]
        y_pred=np.array(y_pred)
        # print percentage of 1s
        #print(sum(y_pred=='1')/len(y_pred))
        train_accuracy+=sum(y_pred==new_y_train)/len(new_y_train)
        w = regression.coef_
        intercept=regression.intercept_
        print(regression.coef_)
        print(regression.intercept_)
        np.save('logistic_weights.npy', w)
        np.save('logistic_bias.npy', intercept)

        # predict the test data
        newdata2 = X_test
        new_y_test = y_test
        # datapoint2=sum(y_test=='1')
        # newdata2=np.vstack((X_test[y_test=='0'][0:datapoint2],X_test[y_test=='1']))
        # newdata2=np.array(newdata2)
        # new_y_test=np.hstack((y_test[y_test=='0'][0:datapoint2],y_test[y_test=='1']))
        # new_y_test=np.array(new_y_test)

        y_testpred=custom_predict(newdata2,0.265,regression)
        y_testpred=[str(y) for y in y_testpred]
        y_testpred=np.array(y_testpred)

        test_accuracy+=sum(y_testpred==new_y_test)/len(new_y_test)
        print(sum(regression.predict(newdata2)=='1')/len(new_y_test))
        print(sum(y_testpred=='1')/len(y_testpred))

        w = regression.coef_
        # saving weights trained by the model.

        # find dominant class in y_train
        # change elements in y_train from string to binary number
        dominant_class='0'
        # change elements in y_train from string to binary number
        y_train_binary=[1 if y=='1' else 0 for y in new_y_train]
        if (sum(y_train_binary)/len(y_train_binary)>0.5):
            dominant_class='1'
        accuracy_choosing_dominant_class+=sum([1 if y==dominant_class else 0 for y in new_y_test])/len(new_y_test)

    print("Test Accuracy: ",test_accuracy/10)
    print("Train Accuracy: ",train_accuracy/10)
    print("Accuracy of choosing dominant class: ",accuracy_choosing_dominant_class/10)
    return train_accuracy/10,test_accuracy/10, accuracy_choosing_dominant_class/10

def number_of_turns(game):
    return len(game)

def import_from_csv_precise(file_path):
    file = open(file_path,encoding='utf-8') # Put the name of the data file. Note: it shall be in the same folder as this file
    csvreader = csv.reader(file)
    header = next(csvreader)
    guesses=[]
    GPT_rating=[]
    for row in csvreader:
        guesses.append(ast.literal_eval(row[2]))
        GPT_rating.append(row[7])
    file.close()
    return guesses,GPT_rating

def test_feature(guesses):
    X=[]
    for i in range(len(guesses)):
        X.append(number_of_turns(guesses[i]))
    return X

def k_nearest_neighbors(X,Y):
    neigh = KNeighborsClassifier(n_neighbors=10,weights='distance')
    train_accuracy=0
    test_accuracy=0
    accuracy_choosing_dominant_class=0
    for i in range(10):
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
        # run the fit function to train the model. 
        # split the data by making sure 1s and 0s of Y are split in 50/50
        datapoint=sum(y_train=='1')
        newdata=np.vstack((X_train[y_train=='0'][0:datapoint],X_train[y_train=='1']))
        newdata=np.array(newdata)
        new_y_train=np.hstack((y_train[y_train=='0'][0:datapoint],y_train[y_train=='1']))
        new_y_train=np.array(new_y_train)
        neigh=neigh.fit(newdata,new_y_train) 
        # run the predict function to get the predictions
        train_accuracy+=neigh.score(newdata,new_y_train)

        # predict the test data
        datapoint2=sum(y_test=='1')
        newdata2=np.vstack((X_test[y_test=='0'][0:datapoint2],X_test[y_test=='1']))
        newdata2=np.array(newdata2)
        new_y_test=np.hstack((y_test[y_test=='0'][0:datapoint2],y_test[y_test=='1']))
        new_y_test=np.array(new_y_test)


        y_pred=neigh.predict(newdata2)
        print(sum(y_pred=='1'))
        test_accuracy+=neigh.score(newdata2,new_y_test)
        # saving weights trained by the model.

        # find dominant class in y_train
        # change elements in y_train from string to binary number
        dominant_class='0'
        # change elements in y_train from string to binary number
        y_train_binary=[1 if y=='1' else 0 for y in new_y_train]
        if (sum(y_train_binary)/len(y_train_binary)>0.5):
            dominant_class='1'
        accuracy_choosing_dominant_class+=sum([1 if y==dominant_class else 0 for y in new_y_test])/len(new_y_test)

    print("Test Accuracy: ",test_accuracy/10)
    print("Train Accuracy: ",train_accuracy/10)
    print("Accuracy of choosing dominant class: ",accuracy_choosing_dominant_class/10)
    return train_accuracy/10,test_accuracy/10, accuracy_choosing_dominant_class/10

# Load the data
#guesses,GPT_rating=import_from_csv_precise('wordle_final_with_GPT_ratings.csv')
#X=test_feature(guesses)
#X = np.array(X)
#np.save('test_feature.npy', X)
#X=np.reshape(X,(-1,1))
#GPT_rating=np.array(GPT_rating)
#train_logistic_regression(X,GPT_rating)
X = np.load('x_embedding_unique.npy')
Y = np.load('GPT_rating_unique.npy')

#print(X[500],Y[500])
#train_logistic_regression(X,Y)
#Y = np.array([1 if y=='1' else 0 for y in Y_before])
train_logistic_regression(X,Y)
#k_nearest_neighbors(X,Y)
