# ---------- Imported libraries ----------

import os
import pandas as pd
import csv


# ---------- Create dataset -----------

def create_dataset_from_directory(directory_source : str,
                                directory_target : str,
                                dataset_name : str) -> None:

    dataset_entries = get_data_set_entries(directory_source)
    df = pd.DataFrame(dataset_entries)
    df.to_csv(f"{directory_target}\\{dataset_name}.csv", index=False, header=False)
    #with open(f"{directory_target}\\{dataset_name}.csv", "w+", errors="ignore") as file:
        #writer = csv.writer(file)
        #writer.writerows(dataset_entries)

def get_data_set_entries(directory_path) -> list:
    files = os.listdir(directory_path)
    dataset_entries = [None]*len(files)

    for i, filename in enumerate(files):
        author, text = get_author_and_text(directory_path + "\\" + filename)
        dataset_entries[i] = [author, text]
    
    return dataset_entries
    

def get_author_and_text(filepath : str) -> list:
    with open(filepath, 'r', errors="ignore") as file:
        lines = file.readlines()
        author = lines[0]
        text = lines[1:]
        #text = f'"{" ".join(lines[1:])}"'

    return [author, text]


source_dir = "C:\\Users\\larsv\\OneDrive\\Documents\\Studier\\Master\\2023 Vår\\Master's thesis\\Unmasking_the_Cheating_Student\\data\\news_articles\\the_sun"
target_dir =  "C:\\Users\\larsv\\OneDrive\\Documents\\Studier\\Master\\2023 Vår\\Master's thesis\\Unmasking_the_Cheating_Student\\data\\datasets"
name = "test"

create_dataset_from_directory(source_dir, target_dir, name)