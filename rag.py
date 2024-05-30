from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma

loader = TextLoader("rag_database.txt")
documents = loader.load()
# 创建拆分器
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=45,
    chunk_overlap=10
)
# 拆分文档
texts = text_splitter.split_documents(documents)

# embedding model: m3e-base
model_name = "moka-ai/m3e-base"
model_kwargs = {'device': 'cuda'}
encode_kwargs = {'normalize_embeddings': True}
embedding = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
    query_instruction="为文本生成向量表示用于文本检索",
)

# 数据入库
db = Chroma.from_documents(texts, embedding)

# 相似度搜索
database = db.similarity_search("屋顶上绘制了瓦片，可能是因为它采用了古代的材料或者设计，以增加瓦片的沉重感。尽管如此，瓦片的线条依然流畅，给人一种简洁而美丽的印象。")

print("\n-".join([a.page_content for a in database]))
