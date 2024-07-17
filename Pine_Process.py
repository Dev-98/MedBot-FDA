import os, random, string
from langchain.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone
from dotenv import load_dotenv
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()
CHUNK_SIZE = 5000
HF_EMBEDDINGS = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
pine = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index = pine.Index(os.environ.get("PINECONE_INDEX"))
genai.configure(api_key=os.environ.get("GENAI_API_KEY"))


def get_context_new(input_query,med_name,k=5):

    input_embed = HF_EMBEDDINGS.embed_query(input_query)

    pinecone_resp = index.query(vector=input_embed, top_k=k, include_metadata=True,
                                filter={"med_intial": med_name,
                                })
    
    if not pinecone_resp['matches']:
        # print(pinecone_resp)
        return "No matches Found, check the metadata "

    context = ""
    for i in range(len(pinecone_resp['matches'])):

        score = pinecone_resp['matches'][i]["score"] 
        print("Score : ", score)
        if score >= 0.53:
            context += "".join(pinecone_resp['matches'][i]['metadata']['text'])
        
    if context == "":
        context = f"No context Found, answer it yourself "
    
    return context


def get_gemini_response(context:str,query:str) -> str :
    input = f""" Your are an expert pharmacist who holds expert level of knowledge of medicines. Use the given context to get the insights of the medicine asked
                and you have to drive your answer accordingly.
          context:{context}
        
          This is the query you have to answer
          query:{query}

        """
    model=genai.GenerativeModel('gemini-pro')
    response=model.generate_content(input)
    return response.text

def upsert_data(medname,text):
    ''' Funtiom to upsert data into Pinecone with company name and text as metadata :
        @param company = name of the company text belongs to
        @param text = text extracted from the company pdf files
        
        @ returns None
    '''

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=20)
    text_chunks = text_splitter.split_text(text)
    print("This the the length of Text chucks : ",len(text_chunks))
    for chunk in text_chunks:
        
        metadata = {"med_intial": medname, "text":chunk}
        # Text to be embedded
        vector = HF_EMBEDDINGS.embed_query(chunk)

        # Ids generation for vectors
        _id = ''.join(random.choices(string.ascii_letters + string.digits, k=10,))

        # Upserting vector into pinecone database
        index.upsert(vectors=[{"id":_id, "values": vector, "metadata": metadata}])

        print("Vector upserted successfully")

if __name__ == "__main__":

    # file_name = "A_med.txt"    
    # with open(file_name, "r") as f:
    #     text = f.read()
    
    # upsert_data("A",text)
    print(get_context_new("What is Phenylephrine and what is the safe dosage of Phenylephrine","A"))
    # print(type(text))
    