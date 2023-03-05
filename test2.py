import concurrent.futures

def get_embedding(text, model="text-embedding-ada-002", max_retries=3):
    text = text.replace("\n", " ")
    for i in range(max_retries):
        try:
            embedding = openai.Embedding.create(input=[text], model=model)['data'][0]['embedding']
            return embedding
        except Exception as e:
            print(f"Error: {e}")
            if i == max_retries - 1:
                raise e

def get_embeddings(texts):
    embeddings = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(get_embedding, text) for text in texts]
        for future in concurrent.futures.as_completed(futures):
            try:
                embedding = future.result()
            except Exception as e:
                print(f"Error: {e}")
            else:
                embeddings.append(embedding)
    return embeddings