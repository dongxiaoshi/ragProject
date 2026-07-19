import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.openai import OpenAI

# 1. 设置环境变量
os.environ["OPENAI_API_KEY"] = os.environ.get("DASHSCOPE_API_KEY", "")

# 2. 如果 DASHSCOPE_API_KEY 没设置，提前报错
if not os.environ["OPENAI_API_KEY"]:
    raise ValueError("请设置 DASHSCOPE_API_KEY 环境变量")

# 3. 关键：配置 LLM 指向阿里云网关
llm = OpenAI(
    model="qwen-plus",  # 或 qwen-turbo, qwen-max
    api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.environ["OPENAI_API_KEY"],
)
# 4. 将 LLM 注入到全局设置
from llama_index.core import Settings
Settings.llm = llm

#步骤一：文档解析
documents= SimpleDirectoryReader("data").load_data()
#步骤二：向量化
index= VectorStoreIndex.from_documents(documents)
#步骤三：查询
query_engine= index.as_query_engine()
#步骤四：得到结果
response= query_engine.query("苹果每斤多少钱?")
print(response)
