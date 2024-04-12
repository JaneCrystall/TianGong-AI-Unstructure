
import pickle

import weaviate
from weaviate.config import AdditionalConfig


def split_chunks(text_list: list, source: str):
    chunks = []
    for text in text_list:
        for key, value in text.items():
            chunks.append({"question": key, "answer": value, "source": source})
    return chunks

with open("pickle/test.pkl", "rb") as f:
    test_list = pickle.load(f)

chunks = split_chunks(test_list, "test")

w_client = weaviate.connect_to_local(
    host="localhost", additional_config=AdditionalConfig(timeout=(60000, 80000))
)


tiangong_colletion = w_client.collections.get(name="tiangong")

for chunk in chunks:
    tiangong_colletion.data.insert(chunk)

# tiangong_colletion.data.insert_many(chunks)
w_client.close()


