import openai
import gitlab
import pandas as pd
import sqlite3
import os
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential

# 1. GitLab credentials and setup
GITLAB_TOKEN = os.getenv('GITLAB_TOKEN')  # Load GitLab token from env variables
GITLAB_URL = 'https://gitlab.com'         # Change this if you're using self-hosted GitLab
GROUP_ID = '<YOUR_GROUP_ID>'              # Replace with your GitLab group ID
chunksize = 1000                          # Number of rows to process at a time

# 2. SQLite Database setup
sqlite_db = 'gitlab_issues.db'  # SQLite database path

# 3. Azure Cognitive Search credentials
AZURE_SEARCH_SERVICE_NAME = '<YOUR_SEARCH_SERVICE_NAME>'  # Replace with your Azure Search service name
AZURE_SEARCH_API_KEY = '<YOUR_SEARCH_API_KEY>'            # Replace with your Azure Search API key
AZURE_SEARCH_INDEX_NAME = '<YOUR_INDEX_NAME>'             # Replace with your Azure Search index name

# 4. OpenAI API credentials
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')  # Load OpenAI API key from env variables
openai.api_key = OPENAI_API_KEY

# 5. Generate Embeddings for each issue using OpenAI API
def generate_embedding(text):
    try:
        response = openai.Embedding.create(input=text, model="text-embedding-ada-002")
        return response['data'][0]['embedding']
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None

# 6. Push data to Azure Cognitive Search (with embeddings)
def push_to_azure_search(documents):
    search_client = SearchClient(
        endpoint=f"https://{AZURE_SEARCH_SERVICE_NAME}.search.windows.net",
        index_name=AZURE_SEARCH_INDEX_NAME,
        credential=AzureKeyCredential(AZURE_SEARCH_API_KEY)
    )

    # Push documents to Azure Search
    result = search_client.upload_documents(documents=documents)
    print(f"Pushed {len(documents)} documents to Azure Search, Result: {result}")

# 7. Fetch issues from GitLab group
def fetch_group_issues(group_id, gitlab_url, gitlab_token):
    # Initialize GitLab connection
    gl = gitlab.Gitlab(gitlab_url, private_token=gitlab_token)
    
    # Get the group and fetch its issues
    group = gl.groups.get(group_id)
    issues = group.issues.list(all=True)
    
    # Convert issues to a list of dictionaries for easier processing
    issue_data = [{
        'id': issue.id,
        'title': issue.title,
        'description': issue.description,
        'state': issue.state,
        'created_at': issue.created_at,
        'updated_at': issue.updated_at,
        'author': issue.author['name'],
        'labels': ','.join(issue.labels),
        'content': f"{issue.title}. {issue.description}",  # Combine title and description for embedding
    } for issue in issues]

    return issue_data

# 8. Store issues in SQLite
def store_issues_in_sqlite(issue_data, sqlite_db):
    conn = sqlite3.connect(sqlite_db)
    df = pd.DataFrame(issue_data)
    df.to_sql('gitlab_issues', conn, if_exists='replace', index=False)
    conn.close()
    print("Issues stored in SQLite")

# 9. Process issues in chunks, generate embeddings, and push to Azure Search
def process_issues_in_chunks_and_embed(sqlite_db, chunksize):
    conn = sqlite3.connect(sqlite_db)
    offset = 0
    
    while True:
        # Fetch a chunk of issues from the SQLite database using LIMIT and OFFSET
        query = f"SELECT * FROM gitlab_issues LIMIT {chunksize} OFFSET {offset}"
        chunk = pd.read_sql(query, conn)
        
        if chunk.empty:
            print("No more issues to process, exiting...")
            break
        
        documents = []
        for _, row in chunk.iterrows():
            # Generate embedding for the issue content (title + description)
            embedding = generate_embedding(row['content'])
            
            if embedding:
                document = {
                    'id': str(row['id']),
                    'title': row['title'],
                    'description': row['description'],
                    'author': row['author'],
                    'state': row['state'],
                    'created_at': row['created_at'],
                    'updated_at': row['updated_at'],
                    'labels': row['labels'],
                    'embedding': embedding  # Store embedding as vector in Azure Search
                }
                documents.append(document)

        if documents:
            # Push the chunk of documents with embeddings to Azure Search
            push_to_azure_search(documents)

        offset += chunksize

    conn.close()

# Main function to tie everything together
def main():
    # Fetch issues from GitLab
    print("Fetching issues from GitLab...")
    issue_data = fetch_group_issues(GROUP_ID, GITLAB_URL, GITLAB_TOKEN)
    
    # Store issues in SQLite
    print("Storing issues in SQLite...")
    store_issues_in_sqlite(issue_data, sqlite_db)
    
    # Process issues in chunks, generate embeddings, and push to Azure Search
    print("Processing issues, generating embeddings, and pushing to Azure Search...")
    process_issues_in_chunks_and_embed(sqlite_db, chunksize)

if __name__ == "__main__":
    main()
