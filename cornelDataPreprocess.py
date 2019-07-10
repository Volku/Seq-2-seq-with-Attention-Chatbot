import codecs
import csv
import os

corpus_name = "cornell movie-dialogs corpus"
corpus = os.path.join(corpus_name)
lines_path = os.path.join(corpus,"movie_lines.txt")
conver_path = os.path.join(corpus,"movie_conversations.txt")


line_fields = ["lineID","characterID","movieID","character","text"]
lines = {}

with open(lines_path, 'r', encoding='iso-8859-1') as f:
    for line in f:
        values = line.split(" +++$+++ ")
        # Extract fields
        lineObj = {}
        for i, field in enumerate(line_fields):
            lineObj[field] = values[i]
        lines[lineObj['lineID']] = lineObj

conv_fields = ["character1ID", "character2ID", "movieID", "utteranceIDs"]
conversations = []

with open(conver_path,'r',encoding='iso-8859-1') as f:
    for line in f.readlines():
        values = line.split('+++$+++')

        convObj = {}
        for i,field in enumerate(conv_fields):
            convObj[field] = values[i]
        lineIds = eval(convObj["utteranceIDs"])

        convObj["lines"] = []
        for lineId in lineIds:
            convObj["lines"].append(lines[lineId])
        conversations.append(convObj)

qa_pairs = []
for conversation in conversations:
    for i in range(len(conversation["lines"])-1):

        inputLine = conversation["lines"][i]["text"].strip()
        targetLine = conversation["lines"][i + 1]["text"].strip()
        # Filter wrong samples (if one of the lists is empty)
        if inputLine and targetLine:
            qa_pairs.append([inputLine, targetLine])

datafile = os.path.join(corpus,"formatted_movie_lines.txt")

delimiter = '\t'

delimiter = str(codecs.decode(delimiter,"unicode_escape"))

#write formatted lines into new csv_file
print("Writing new formatted file. please w8")
with open(datafile, 'w', encoding='utf-8') as outputfile:
    writer = csv.writer(outputfile, delimiter=delimiter)
    for pair in qa_pairs:
        writer.writerow(pair)



