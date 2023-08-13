import pandas as pd
from sentence_transformers import SentenceTransformer
import torch
import pinecone
from transformers import pipeline, TapasTokenizer, TapasForQuestionAnswering
from tqdm.auto import tqdm
from datasets import load_dataset

# load the dataset from huggingface datasets hub
data = load_dataset("ashraq/ott-qa-20k", split="train")

print(data[2])
# Load data from your Excel file
import pandas as pd

# store all tables in the tables list
tables = []
# loop through the dataset and convert tabular data to pandas dataframes
for doc in data:
    table = pd.DataFrame(doc["data"], columns=doc["header"])
    tables.append(table)
print(tables[2])
# set device to GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# load the table embedding model from huggingface models hub
retriever = SentenceTransformer("deepset/all-mpnet-base-v2-table", device=device)
print(retriever)

def _preprocess_tables(tables: list):
    processed = []
    for table in tables:
        processed_table = "\n".join([table.to_csv(index=False)])
        processed.append(processed_table)
    return processed

# Format all the dataframes in the tables list
processed_tables = _preprocess_tables(tables)
print(processed_tables[2])
# Connect to Pinecone environment
pinecone.init(
    api_key="83af22f9-ec60-484a-8eba-3519c69b251f",
    environment="eu-west4-gcp"
)

# you can choose any name for the index
index_name = "table-qa"

# check if the table-qa index exists
if index_name not in pinecone.list_indexes():
    # create the index if it does not exist
    pinecone.create_index(
        index_name,
        dimension=768,
        metric="cosine"
    )

# connect to table-qa index we created
index = pinecone.Index(index_name)


# we will use batches of 64
batch_size = 64

for i in tqdm(range(0, len(processed_tables), batch_size)):
    # find end of batch
    i_end = min(i+batch_size, len(processed_tables))
    # extract batch
    batch = processed_tables[i:i_end]
    # generate embeddings for batch
    emb = retriever.encode(batch).tolist()
    # create unique IDs ranging from zero to the total number of tables in the dataset
    ids = [f"{idx}" for idx in range(i, i_end)]
    # add all to upsert list
    to_upsert = list(zip(ids, emb))
    # upsert/insert these records to pinecone
    _ = index.upsert(vectors=to_upsert)

query = "which country has the highest GDP in 2020?"
# generate embedding for the query
xq = retriever.encode([query]).tolist()
# query pinecone index to find the table containing answer to the query
result = index.query(xq, top_k=1)
print(result)
id = int(result["matches"][0]["id"])
print(tables[id].head())

from transformers import pipeline, TapasTokenizer, TapasForQuestionAnswering

model_name = "google/tapas-base-finetuned-wtq"
# load the tokenizer and the model from huggingface model hub
tokenizer = TapasTokenizer.from_pretrained(model_name)
model = TapasForQuestionAnswering.from_pretrained(model_name, local_files_only=False)
# load the model and tokenizer into a question-answering pipeline
pipe = pipeline("table-question-answering",  model=model, tokenizer=tokenizer, device=device)
print(pipe(table=tables[id], query=query))

def query_pinecone(query):
    # generate embedding for the query
    xq = retriever.encode([query]).tolist()
    # query pinecone index to find the table containing answer to the query
    result = index.query(xq, top_k=1)
    # return the relevant table from the tables list
    return tables[int(result["matches"][0]["id"])]
def get_answer_from_table(table, query):
    # run the table and query through the question-answering pipeline
    answers = pipe(table=table, query=query)
    return answers
query = "which car manufacturers produce cars with a top speed of above 180 kph?"
table = query_pinecone(query)
print(table)