
from langchain_community.document_loaders import PyPDFLoader
import re
from langchain.schema import Document
import pdfplumber
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
# from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma.vectorstores import Chroma

# load pdf
pdf_path = "phy_book.pdf"
if pdf_path:
    loader = PyPDFLoader(file_path=pdf_path)
    data = loader.load()
else:
    print("PDF not found")
    exit()

print("pdf loaded")

# Remove extra whitespace
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text

cleaned_data = []
for doc in data:
    cleaned_data.append(Document(page_content=clean_text(doc.page_content), metadata=doc.metadata))

print("removed extra whitespaces")

# handle math exprs
patterns = [
    r'\b\d+(\.\d+)?\s*[-+*/^]\s*\d+(\.\d+)?\b',  # Basic arithmetic operations (e.g., 3 + 5, 4.2 * 7)
    r'[A-Za-z]+\s*=\s*[\dA-Za-z+\-*/^()]+',       # Simple equations (e.g., x = 2 + 3)
    r'\b\d+\s*[+\-*/^()]+\s*\d+',                 # Algebraic expressions (e.g., 2 * (3 + 4))
    r'\b[A-Za-z]+\s*\d*[_^]\d+\b',                # Variables with subscripts or superscripts (e.g., x_2, y^2)
    r'\b\w+\s*\(.*?\)\b',                         # Function calls (e.g., sin(x), log(2))
    r'\b\w+\s*=\s*\w+\s*[-+*/]\s*\w+',            # Equations with variable operations (e.g., y = x + z)
    r'\d+(\.\d+)?\s*[-+*/^()]+\d+',               # Numeric expressions (e.g., 5^2, (3.14 * 2) + 4)
    r'\b\w+\s*\(\s*\w+\s*\)\s*=\s*[\w\d+\-*/^()]+',# Function definitions (e.g., f(x) = x^2 + 2x)
    r'[\dA-Za-z]+\s*=\s*[\dA-Za-z+\-*/^()]+',      # General form of equations (e.g., E = mc^2)
]

def expr_cleanup(math_expressions):
    # Remove duplicates by converting the list to a set
    unique_expressions = list(set(math_expressions))
    
    # Sort by length to filter out partial expressions
    unique_expressions.sort(key=lambda x: len(x), reverse=True)
    
    # Remove partial matches
    final_expressions = []
    for i, expr in enumerate(unique_expressions):
        # Ensure we don't include plain text or partial expressions
        if not any(expr in larger_expr for larger_expr in unique_expressions[:i]):
            # Check for presence of mathematical symbols to remove plain text
            if re.search(r'[=+\-*/^]', expr):
                final_expressions.append(expr)
    return final_expressions

def extract_math_expressions(text):
    math_expressions = []
    for pattern in patterns:
        matches = re.findall(pattern, text)
        # math_expressions.extend([match.strip() for match in matches])
        math_expressions.extend([match if isinstance(match, str) else ''.join(match) for match in matches])
    return expr_cleanup(math_expressions)

# Extract matching patterns from pdf
for doc in cleaned_data:
    math_exprs = extract_math_expressions(doc.page_content)
    
    if isinstance(math_exprs, list):
        math_exprs = "; ".join(math_exprs)  # Join the list elements into a single string
        
    doc.metadata['math_expressions'] = math_exprs

print("handled math expressions")

# Process Tables
tables = []
with pdfplumber.open(pdf_path) as pdf:
    for page in pdf.pages:
        table = page.extract_table()
        if table:
            tables.append(table)

# Convert tables to DataFrames
table_dfs = [pd.DataFrame(table[1:], columns=table[0]) for table in tables]

# Function to clean up columns: remove empty columns and make names unique
def clean_dataframe(df):
    # Remove empty or None columns
    df = df.loc[:, df.columns.notnull()]  # Keep columns that are not None
    df = df.loc[:, df.columns != '']  # Keep columns that are not empty strings
    
    # Check for duplicates
    if not df.columns.is_unique:
        print(f"Duplicate columns found in DataFrame:\n{df.columns[df.columns.duplicated()]}")
        
    # Make columns unique
    cols = pd.Series(df.columns)
    for dup in cols[cols.duplicated()].unique():  # Find duplicates
        cols[cols[cols == dup].index.values.tolist()] = [dup + '_' + str(i) if i != 0 else dup for i in range(sum(cols == dup))]
    df.columns = cols
    
    return df

# Function to ensure unique column names
def make_columns_unique(df):
    if not df.columns.is_unique:
        print(f"Duplicate columns found in DataFrame:\n{df.columns[df.columns.duplicated()]}")
        
    cols = pd.Series(df.columns)
    for dup in cols[cols.duplicated()].unique():  # Find duplicates
        cols[cols[cols == dup].index.values.tolist()] = [dup + '_' + str(i) if i != 0 else dup for i in range(sum(cols == dup))]
    df.columns = cols
    return df

# Convert DataFrames to JSON
table_jsons = []
for df in table_dfs:
    df_cleaned = clean_dataframe(df)  # Clean the DataFrame
    # print(f"Unique columns for cleaned DataFrame:\n{df_cleaned.columns}")
    # Convert to JSON only if the DataFrame is not empty
    if not df_cleaned.empty:
        json_str = df_cleaned.to_json(orient='records')
        table_jsons.append(json_str)  # Append only non-empty JSON strings

# Create documents for the tables
table_data = [Document(page_content=text, metadata={'source': 'table', 'page': i, 'type': "table"}) for i, text in enumerate (table_jsons)]

print("processed tables")

# combine table data with pdf text contents
combined_data = cleaned_data + table_data

print("combined_data created")


# Split and chunk

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000, 
    chunk_overlap = 100,
    length_function = len,
)
chunked_document = text_splitter.split_documents(combined_data)

print("document chunked")

# Embed the chunked document using OllamaEmbeddings (nomic-embed-text)
embedding_model = OllamaEmbeddings(model="nomic-embed-text")

# # Embed the chunked document using SentenceTransformer (all-MiniLM-L6-v2)
# embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Create the Chroma vector store
try:
    vector_db = Chroma.from_documents(
        documents=chunked_document,
        embedding=embedding_model,
        collection_name="local-rag",
        persist_directory="./dtbs/db_nomic"
    )
    print("Embedded Documents stored in ChromaDB successfully!")
except Exception as e:
    print(f"An error occurred: {e}")
    
print("finish")