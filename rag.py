import os
import json
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from embedding import get_embedding_function
from groq import Groq

os.environ['GROQ_API_KEY'] = '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'
CHROMA_PATH = "chroma2"
CONVERSATION_HISTORY_FILE = "conversation_history.json"

PROMPT_TEMPLATE = """
Based on the following context, answer the question concise:
patient details:
[
Status: pregnant
Trimester: 2nd
Symptoms: yes
Support: no
]   

Context:
{context_text}
{context_text2}

---

Answer this question: {question}
"""

def load_conversation_history():
    if os.path.exists(CONVERSATION_HISTORY_FILE):
        try:
            with open(CONVERSATION_HISTORY_FILE, 'r') as file:
                return json.load(file)
        except (json.JSONDecodeError, ValueError):
            print("Warning: Conversation history file is corrupted or empty. Starting fresh.")
            return []  # Return an empty list if the file is invalid
    return []  # Return an empty list if the file does not exist

def save_conversation_history(history):
    with open(CONVERSATION_HISTORY_FILE, 'w') as file:
        json.dump(history, file)

def query_retro_rag(query_text: str) -> str:
    # Load existing conversation history
    conversation_history = load_conversation_history()

    # Step 1: First retrieval pass based on the initial query
    embedding_function = get_embedding_function()

    # Truncate the query if it's too long
    if len(query_text.split()) > 512:  # Adjust this to your token limit
        query_text = ' '.join(query_text.split()[:512])  # Keep only the first 512 words

    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    results_first_pass = db.similarity_search_with_score(query_text, k=5)

    # Combine the context from the first retrieval
    context_text_1 = "\n\n---\n\n".join([doc.page_content for doc, _ in results_first_pass])

    # Append the user's query to the conversation history
    conversation_history.append({"role": "user", "content": query_text})

    # Construct the full conversation history string
    full_conversation = "\n".join([f"{item['role']}: {item['content']}" for item in conversation_history])

    # Step 2: Generate initial response based on the first retrieval context
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    initial_prompt = prompt_template.format(
        context_text=context_text_1,
        context_text2="",  # Initially empty for the second pass
        question=query_text
    )

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    initial_chat_completion = client.chat.completions.create(
        messages=[{"role": "system", "content": "You are an AI-powered pregnancy care chatbot. Ask the user how many months they are into their pregnancy, inquire about any inconveniences they are facing, and provide diet plan suggestions and relevant advice. If any other questions are asked,keep it concise"}] + conversation_history,
        model="llama-3.1-70b-versatile",
        temperature=0.5,
        top_p=0.90,
    )

    initial_response_text = initial_chat_completion.choices[0].message.content

    # Step 3: Use the initial response to perform a second retrieval (refining the query)
    refined_query = query_text + " " + initial_response_text  # Combine query and response
    if len(refined_query.split()) > 512:
        refined_query = ' '.join(refined_query.split()[:512])  # Truncate if necessary

    results_second_pass = db.similarity_search_with_score(refined_query, k=5)

    # Combine the context from the second retrieval
    context_text_2 = "\n\n---\n\n".join([doc.page_content for doc, _ in results_second_pass])

    # Step 4: Generate the final response based on both retrieval contexts
    final_prompt = prompt_template.format(
        context_text=context_text_1,  # Context from the first pass
        context_text2=context_text_2,  # Additional context from the second pass
        question=query_text
    )

    final_chat_completion = client.chat.completions.create(
        messages=[{"role": "system", "content": "You are an AI-powered pregnancy care chatbot. Ask the user how many months they are into their pregnancy, inquire about any inconveniences they are facing, and provide diet plan suggestions and relevant advice. If any other questions are asked, do not answer and don't recommend specific clinics."}] + conversation_history,
        model="llama-3.1-70b-versatile",
        temperature=0.4,
        top_p=0.90,
    )

    final_response_text = final_chat_completion.choices[0].message.content

    # Step 5: Update conversation history and save it
    conversation_history.append({"role": "system", "content": final_response_text})
    save_conversation_history(conversation_history)  # Save history after each interaction

    # Return the final formatted response
    return final_response_text

# Main logic for text-based chatbot
if __name__ == "__main__":
    print("Hello, how can I assist you today?")
    
    while True:
        user_input = input("User: ")
        if user_input:
            response = query_retro_rag(user_input)
            print(f"Chatbot: {response}")
