import sys
import csv
import random

# Pass percentage and source CSV file
# A CSV file is generated with random rows of the source file

percentage = float(sys.argv[1]) / 100
path = sys.argv[2]
selected_rows = []

with open(path, "rb") as csv_file:
    reader = csv.reader(csv_file, delimiter=';')
    rows = []
    for row in reader:
        rows.append(row)

selected_rows.append(rows[0])
del rows[0]
n = len(rows)

while(len(selected_rows) < percentage * n):
    chosen = random.choice(rows)
    selected_rows.append(chosen)
    del rows[rows.index(chosen)]
    print(chosen)

with open("sample_" + str(percentage) + ".csv", "wb") as selected_file:
    processed_writer = csv.writer(
        selected_file, delimiter=';', lineterminator='\r\n',
        quoting=csv.QUOTE_NONE)
    for row in selected_rows:
        processed_writer.writerow(row)
