import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core import Settings
from llama_index.llms.dashscope import DashScope
from llama_index.embeddings.dashscope import DashScopeEmbedding

# 1. 配置阿里云
api_key = os.environ.get("DASHSCOPE_API_KEY", "")
if not api_key:
    raise ValueError("请设置 DASHSCOPE_API_KEY 环境变量")

# 2. 配置 LLM（DashScope 原生）
llm = DashScope(
    model_name="qwen-plus",
    api_key=api_key,
)

# 3. 配置 Embedding（DashScope 原生）
embed_model = DashScopeEmbedding(
    model_name="text-embedding-v2",
    api_key=api_key,
)

# 4. 注入全局设置
Settings.llm = llm
Settings.embed_model = embed_model

# 5. 正常使用
documents = SimpleDirectoryReader("data").load_data()
print(f"✅ 成功加载 {len(documents)} 个文档")


index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()
response = query_engine.query("苹果每斤多少钱?")
print(f"\n📝 回答：{response}")
