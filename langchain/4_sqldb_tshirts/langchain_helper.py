import os

import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import SQLDatabase
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from langchain_experimental.sql import SQLDatabaseChain
from langchain.prompts import (
    SemanticSimilarityExampleSelector,
    FewShotPromptTemplate,
)
from langchain.prompts.prompt import PromptTemplate
from langchain.chains.sql_database.prompt import PROMPT_SUFFIX

from few_shots import few_shots


# Load environment variables from the same folder as this file
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))


@st.cache_resource(show_spinner="Connecting to database and loading AI models...")
def get_few_shot_db_chain():

    # 1. MySQL configuration
    db_user = os.getenv("DB_USER")
    db_password = os.getenv("DB_PASSWORD")
    db_host = os.getenv("DB_HOST", "localhost")
    db_name = os.getenv("DB_NAME", "atliq_tshirts")

    # 2. Google API key
    google_api_key = os.getenv("GOOGLE_API_KEY")

    required_variables = {
        "DB_USER": db_user,
        "DB_PASSWORD": db_password,
        "GOOGLE_API_KEY": google_api_key,
    }

    missing = [
        name for name, value in required_variables.items()
        if not value
    ]

    if missing:
        raise ValueError(
            f"Missing environment variables: {', '.join(missing)}. "
            "Please check your .env file."
        )

    # 3. Connect to MySQL
    # URL.create safely handles special characters in passwords
    connection_url = URL.create(
        drivername="mysql+pymysql",
        username=db_user,
        password=db_password,
        host=db_host,
        database=db_name,
    )

    engine = create_engine(connection_url)

    db = SQLDatabase(
        engine,
        sample_rows_in_table_info=3
    )

    # 4. Initialize Gemini
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
        google_api_key=google_api_key,
        temperature=0.2,
    )

    # 5. Initialize HuggingFace embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # 6. Prepare few-shot examples
    to_vectorize = [
        " ".join(str(value) for value in example.values())
        for example in few_shots
    ]

    # 7. Initialize Chroma vector database
    vectorstore = Chroma.from_texts(
        texts=to_vectorize,
        embedding=embeddings,
        metadatas=few_shots,
        collection_name="atliq_tshirts_examples",
        persist_directory="./chroma_db",
    )

    # 8. Select similar examples
    example_selector = SemanticSimilarityExampleSelector(
        vectorstore=vectorstore,
        k=2,
    )

    # 9. MySQL prompt
    mysql_prompt = """
You are a MySQL expert.

Given an input question, first create a syntactically correct
MySQL query to run, then look at the results of the query and
return the answer to the input question.

Unless the user specifies a specific number of examples,
query for at most {top_k} results using LIMIT.

Never query for all columns from a table.
Only query the columns needed to answer the question.

Wrap each column name in backticks (`).

Use only the column names present in the database schema.

Pay attention to which column belongs to which table.

Use CURDATE() when the question involves today.

Only generate read-only SELECT queries.
Never modify, delete, or insert database records.

Use the following format:

Question: Question here
SQLQuery: Query to run with no preamble
SQLResult: Result of the SQLQuery
Answer: Final answer here

No preamble.
"""

    # 10. Example prompt
    example_prompt = PromptTemplate(
        input_variables=[
            "Question",
            "SQLQuery",
            "SQLResult",
            "Answer",
        ],
        template="""
Question: {Question}
SQLQuery: {SQLQuery}
SQLResult: {SQLResult}
Answer: {Answer}
""",
    )

    # 11. Few-shot prompt template
    few_shot_prompt = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=example_prompt,
        prefix=mysql_prompt,
        suffix=PROMPT_SUFFIX,
        input_variables=[
            "input",
            "table_info",
            "top_k",
        ],
    )

    # 12. Create SQL database chain
    chain = SQLDatabaseChain.from_llm(
        llm=llm,
        db=db,
        prompt=few_shot_prompt,
        verbose=True,
        return_intermediate_steps=False,
    )

    return chain