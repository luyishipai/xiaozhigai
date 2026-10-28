
import os
import logging
from functools import lru_cache

import dotenv
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    PyPDFLoader, Docx2txtLoader, TextLoader, UnstructuredMarkdownLoader,
)

logger = logging.getLogger(__name__)

# ─── 绝对路径 ─────────────────────────────────────────────
_AGENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_AGENT_DIR))  # xiaozhi-server/
_TRAVEL_DB_PATH = os.path.join(_AGENT_DIR, "travel.db")
_VECTORSTORE_DIR = os.path.join(_AGENT_DIR, "data", "vectorstore")
_ENV_FILE = os.path.join(_PROJECT_ROOT, "config", "agent.env")

# ─── 加载环境变量 ─────────────────────────────────────────
dotenv.load_dotenv(_ENV_FILE)

# ─── SQL 数据库 ───────────────────────────────────────────
db = SQLDatabase.from_uri(f"sqlite:///{_TRAVEL_DB_PATH}")

# ─── SQL 生成 LLM（qwen-plus）────────────────────────────
llm = ChatOpenAI(
    model="qwen-plus",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.environ.get("qwen_api_key"),
)

query_prompt_template = ChatPromptTemplate.from_messages([
    ("system", """你是一位{dialect}专家。仅处理身份声明类输入，
身份声明类输入例如「我是小明」「我叫小红」「我的名字是XX」，
自动提取名字，生成匹配该用户画像的SQL，无需用户主动说「查询」

除非特别说明，始终将查询结果限制在最多{top_k}个。
切勿查询所有列，仅请求相关列，但 user_id 列必须始终包含在 SELECT 中，用于后续关联查询。
执行前检查SQL语法，出错时重写重试。

如果问题似乎与数据库无关或者不是表明身份的语句，只需返回""。

仅使用以下表格：
{table_info}

输入内容：{input}""")
])


def search_database(user_input: str):
    """根据用户身份声明自动生成SQL并返回原始元组列表。"""
    prompt = query_prompt_template.invoke({
        "dialect": db.dialect,
        "top_k": 10,
        "table_info": db.get_table_info(),
        "input": user_input,
    })
    query = llm.invoke(prompt).content
    if not query or query.strip() in ['""', '"', 'None']:
        return "无用户信息"
    return db.run(query)


def update_database(sql: str):
    """执行 INSERT/UPDATE/DELETE 等写操作 SQL。"""
    try:
        db.run(sql)
        logger.info(f"[DB写入成功] {sql}")
    except Exception as e:
        logger.error(f"[DB写入失败] {e}")


# ─── 向量数据库 ───────────────────────────────────────────
embeddings = DashScopeEmbeddings(
    model="text-embedding-v2",
    dashscope_api_key=os.getenv("qwen_api_key"),
)
_vectorstore_cache = {}


def _init_vectorstore():
    """扫描 data/vectorstore/ 目录，把所有已有向量库一次性加载到内存"""
    if not os.path.exists(_VECTORSTORE_DIR):
        logger.warning(f"向量库目录不存在: {_VECTORSTORE_DIR}")
        return
    for name in os.listdir(_VECTORSTORE_DIR):
        path = os.path.join(_VECTORSTORE_DIR, name)
        if os.path.isdir(path):
            try:
                _vectorstore_cache[name] = Chroma(
                    persist_directory=path,
                    embedding_function=embeddings,
                    collection_name="xiangliangku",
                )
                logger.info(f"预加载向量库 user_id={name}")
            except Exception as e:
                logger.warning(f"加载 user_id={name} 失败: {e}")


_init_vectorstore()


def load_document(file_path):
    """根据文件类型加载文档"""
    file_path = os.path.normpath(file_path)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")
    file_ext = os.path.splitext(file_path)[1].lower()
    if file_ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif file_ext == ".docx":
        loader = Docx2txtLoader(file_path)
    elif file_ext == ".txt":
        loader = TextLoader(file_path, encoding='utf-8')
    elif file_ext == ".md":
        loader = UnstructuredMarkdownLoader(file_path)
    else:
        raise ValueError(f"不支持的文件类型: {file_ext}")
    return loader.load()


def process_document(file_path, user_id):
    """处理文档并存入向量数据库"""
    try:
        documents = load_document(file_path)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=250, chunk_overlap=25,
            separators=['\n', '。', '；', ''],
        )
        splits = text_splitter.split_documents(documents)
        vectorstore_path = os.path.join(_VECTORSTORE_DIR, str(user_id))
        os.makedirs(vectorstore_path, exist_ok=True)
        Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=vectorstore_path,
            collection_name="xiangliangku",
        )
        logger.info(f"向量存储已保存到 {vectorstore_path}")
        return True
    except Exception as e:
        logger.error(f"文档处理错误: {str(e)}")
        return False


@lru_cache(maxsize=1)
def get_vectorstore(user_id):
    """获取用户的向量存储（直接返回全局预加载的单例）"""
    return _vectorstore_cache.get(str(user_id))


def search_documents(user_id, query, k=3):
    """在用户文档中搜索相关内容"""
    vectorstore = get_vectorstore(user_id)
    if vectorstore is None:
        return []
    return vectorstore.similarity_search(query, k=k)
