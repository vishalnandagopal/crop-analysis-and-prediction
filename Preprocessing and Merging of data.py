import csv
from hashlib import sha256
from os import listdir
from time import sleep

filenames = [file for file in listdir() if file.endswith(".csv") and file != "data.csv"]
defaults = [0]

data: list[list] = []

if len(filenames):
    with open(filenames.pop(0), "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        for row in csv_reader:
            data.append(row)


def id_getter(row: list) -> tuple[int]:
    id1 = row.index("Year")
    id2 = row.index("State Name")
    id3 = row.index("Dist Name")
    return (id1, id2, id3)


def unique_id_maker(row, id1, id2, id3):
    return sha256(
        bytes(str(row[id1]) + str(row[id2]) + str(row[id3]), "utf8")
    ).hexdigest()


id1, id2, id3 = id_getter(data[0])
data[0].insert(0, "Unique ID")

for row in data[1:]:
    row.insert(0, unique_id_maker(row, id1, id2, id3))

id_column = [row[0] for row in data]

for i, file in enumerate(filenames):
    with open(file, "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        for row in csv_reader:
            headers = row
            break
        id1, id2, id3 = id_getter(headers)
        for i, header in enumerate(headers):
            _, __ = header, i
            if header not in data[0]:
                with open(file, "r") as csv_file_local:
                    csv_reader_local = csv.reader(csv_file_local, delimiter=",")
                    print(f"Adding header {header}")
                    sleep(5)
                    data[0].append(header)
                    for row in csv_reader_local:
                        obj = row[i]
                        uniq_id = unique_id_maker(row, id1, id2, id3)
                        for data_row in data[1:]:
                            if data_row[0] == uniq_id:
                                # print(f"Appending {row[i]} for {header} at {data_row[0]}")
                                data_row.append(row[i])
                                break
                    length = len(data[0])
                    for data_row in data[1:]:
                        if len(data_row) < length:
                            data_row.append(0)

with open("./datasets/data.csv", "w") as f:
    csv_writer = csv.writer(f)
    for row in data:
        csv_writer.writerow(row)
    print("Written to file")
