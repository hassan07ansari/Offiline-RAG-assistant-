import ollama

def generate_embedding(text:str):
    response = ollama.embeddings(
        model = "nomic-embed-text",
        prompt = text
    )

    return response["embedding"]


def generate_embeddings(texts,batch_size = 8):
    all_embeddings = []

    for i in range(0,len(texts),batch_size): #generating batch starting postions 

        batch = texts[i:i + batch_size] #extracting one batch

        response = ollama.embed(
            model = "nomic-embed-text",
            input = batch
        )

        all_embeddings.extend(response["embeddings"]) #adding to main list all the vectors

    return all_embeddings



