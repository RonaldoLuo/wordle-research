import csv

def import_from_csv_data(file_path):
    file = open(file_path,encoding='utf-8') # Put the name of the data file. Note: it shall be in the same folder as this file
    csvreader = csv.reader(file)
    header = next(csvreader)
    data=[]
    for row in csvreader:
        data.append(row)
    file.close()
    return data

data=import_from_csv_data("wordle_funny_humor_new_heading.csv")
print(data[0])