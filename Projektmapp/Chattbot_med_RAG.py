

"""
En enkel RAG-baserad chattbot som läser in en PDF-fil.
Boten svarar endast baserat på innehållet i Lightnovel.pdf.
"""

"""
pip install langchain langchain-text-splitters langchain-community bs4
pip install -U "langchain[google-genai]
pip install -qU langchain-google-genai
pip install -qU langchain-community faiss-cpu

"""

import os
from langchain.chat_models import init_chat_model
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools import tool
from langchain.agents import create_agent

# 1. Konfiguration
os.environ["GOOGLE_API_KEY"] = "DIN_GOOGLE_API_NYCKEL"  # Byt ut mot din Google API-nyckel

# Initiera modell (gemini-2.0-flash) Ifall det behövs, byt till annan modell
model = init_chat_model("google_genai:gemini-2.0-flash")

# 2. Embeddings för att omvandla text till vektorer
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004") 

# 3. LADDA PDF-FILEN
loader = PyPDFLoader("Lightnovel.pdf")
docs = loader.load()

# 4. Dela upp PDF-texten i hanterbara bitar (för bättre sökbarhet)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=200,
    add_start_index=True
)
all_splits = text_splitter.split_documents(docs) # Dela upp alla dokument

# 5. Skapa sökbar databas (Vektorstore) från PDF-innehållet 
vector_store = FAISS.from_documents(documents=all_splits, embedding=embeddings) # Vi använder FAISS som vektorstore

# 6. Definiera verktyget för att hämta info från PDF:en
@tool
def retrieve_pdf_context(query: str) -> str:
    """Används för att hitta information i den uppladdade PDF-filen."""
    retrieved_docs = vector_store.similarity_search(query, k=3) # Hämta de 3 mest relevanta dokumenten
    return "\n\n".join([doc.page_content for doc in retrieved_docs]) # Kombinera innehållet från de hämtade dokumenten

# 7. Skapa agenten med strikta instruktioner
tools = [retrieve_pdf_context]
prompt = (
    "Du är en assistent som enbart svarar baserat på innehållet i den bifogade PDF-filen. "
    "Använd verktyget retrieve_pdf_context för att hitta svar. "
    "Om informationen inte finns i PDF-filen, säg att du inte vet."
)

agent = create_agent(model, tools, system_prompt=prompt) # Skapa agenten med verktyg och prompt

# 8. Testa boten
query = "Vem är Han Li och vad kallas han i byn?" # Exempelfråga (Bör få fram: "Han Li är huvudpersonen i berättelsen och kallas 'Second fool' i byn etc...")

for event in agent.stream({"messages": [{"role": "user", "content": query}]}): # Strömma svaret
    if "messages" in event: # Kolla om det finns meddelanden i eventet
        event["messages"][-1].pretty_print() # Skriv ut det senaste meddelandet från agenten



"""
-------------------------------------------------
Möjlig användning i verkligheten:
Denna chattbot kan användas för att ställa frågor om en bok,
manual eller annan dokumentation utan att behöva läsa allt själv.

Utmaningar och möjligheter:
- Långa PDF-filer kan kräva chunking för bättre precision.
- Risk för hallucinationer minskar genom RAG.
- Upphovsrättsliga frågor kan uppstå vid kommersiell användning.

Tekniskt utveckling:
- Citeringar: Att modellen svarar "Enligt sidan 42".
- Användargränssnitt (GUI): Bygga en webbapp med Streamlit eller React.
-------------------------------------------------
"""
