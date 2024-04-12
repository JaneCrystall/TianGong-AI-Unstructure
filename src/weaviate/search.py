import weaviate
from weaviate.classes.query import Rerank
from weaviate.config import AdditionalConfig

w_client = weaviate.connect_to_local(
    host="localhost", additional_config=AdditionalConfig(timeout=(600, 800))
)

content = w_client.collections.get("Tiangong")
query = """船舶污染物排放标准"""
response = content.query.near_text(
    query=query,
    limit=2,
    target_vector="answer",
    rerank=Rerank(prop="question", query="publication"),
)
print(response)
