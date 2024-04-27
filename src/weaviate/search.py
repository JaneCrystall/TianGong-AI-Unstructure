import weaviate
from weaviate.classes.query import Rerank
from weaviate.config import AdditionalConfig

w_client = weaviate.connect_to_local(
    host="localhost", port=8088, additional_config=AdditionalConfig(timeout=(600, 800))
)

content = w_client.collections.get("Tiangong")
query = """项目建设必要性是什么？"""
response = content.query.near_text(
    query=query,
    limit=10,
    target_vector="content",
)
print(response)
