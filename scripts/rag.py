from langchain_community.document_loaders import TextLoader

def load_txt_files(file_path):
    loader = TextLoader(file_path)

    docs = loader.load()
    return docs

if __name__ == "__main__":
    file_path = "input_data/concept_net.txt"
    docs = load_txt_files(file_path)
    print(docs[0].metadata)