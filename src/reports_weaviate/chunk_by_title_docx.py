import concurrent.futures
import os
import pickle

from dotenv import load_dotenv
from tools.unstructure_docx import unstructure_word


load_dotenv()


def process_pdf(file_name):
    # record_id = record["id"]

    text_list = unstructure_word("download/" + file_name + ".docx")

    with open("pickle/"+ filename + ".pkl", "wb") as f:
        pickle.dump(text_list, f)



# record = {"id": "rec_clu17n8bslsq4fnfc8s0"}


filename = "gd"

process_pdf(filename)

print("done")
# for record in records:
#     process_pdf(record)

# with concurrent.futures.ProcessPoolExecutor(max_workers=30) as executor:
#     executor.map(process_pdf, records)
