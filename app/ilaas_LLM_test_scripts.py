# ########## PRINTS ALL AVAILABLE LLM MODELS
# import urllib3
# from config import client
# urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# prompt = "je suis enceinte et je suis très inquiète. comment me préparer pour acceuillir mon bébé ?"

# models = client.models.list().data
# print(f"\nFound {len(models)} models\n")

# for model in models:
#     model_name = model.id
#     print("=" * 50)
#     print(f"Model: {model_name}")

#     try:
#         response = client.chat.completions.create(
#             model=model_name,
#             messages=[{"role": "user", "content": prompt}],
#             max_tokens=200
#         )
#         answer = response.choices[0].message.content
#         print("Response:")
#         print(answer)
 
#     except Exception as e:
#         print(f"Error with model {model_name}: {e}")





########## PRINTS ALL AVAILABLE LLM MODELS
from sentence_transformers import SentenceTransformer
import urllib3
from config import client
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

models = client.models.list()
for m in models.data:
    print(m.id)





########## PRINTS 10 VECTORS FROM EMBEDDINGS
# from sentence_transformers import SentenceTransformer
# import urllib3
# urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# model = SentenceTransformer("BAAI/bge-m3")
# text = "Bonjour, comment allez-vous ?"
# embedding = model.encode(text)
# print(len(embedding))
# print(embedding[:10])