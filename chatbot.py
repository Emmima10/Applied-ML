from transformers import pipeline
import streamlit as st

# Step 1: Your knowledge base
knowledge_base = """
- The West region had the highest total sales.
- Technology category was the most profitable.
- Phones were the top-selling sub-category.
- Discounts above 30% caused major profit reductions.
- XGBoost model achieved RMSE: $39,000 and MAPE: 13.8%.
- Random Forest model achieved RMSE: $42,000 and MAPE: 16.5%.
"""

# Step 2: Load a Question-Answering pipeline
qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-small")

# Step 3: Define the chatbot function for both terminal + web
def get_answer(user_input):
    prompt = f"Based on the following information: {knowledge_base} \nAnswer this question: {user_input}"
    response = qa_pipeline(prompt, max_length=200, truncation=True)
    answer = response[0]['generated_text']
    return answer

# Step 4A: Terminal Version (Optional)
def huggingface_qa_chatbot_terminal():
    print("🧠 Welcome to the FLAN-T5 Sales Insights Chatbot!")
    print("Ask me anything about sales, models, products. Type 'exit' to quit.")
    
    while True:
        user_input = input("\nYou: ")
        
        if user_input.lower() == "exit":
            print("Bot: Goodbye!")
            break
        
        answer = get_answer(user_input)
        print(f"Bot: {answer}")

# Step 4B: Streamlit Version (Web App)
def huggingface_qa_chatbot_streamlit():
    st.set_page_config(page_title="Sales Insights Chatbot 💬", page_icon="💬")

    st.title("📊 Sales Insights Chatbot")
    st.write("Ask me anything about sales, models, or products based on EDA and modeling insights!")

    user_input = st.text_input("Your Question:")

    if user_input:
        with st.spinner("Thinking..."):
            answer = get_answer(user_input)
        st.success(f"**Bot:** {answer}")

# Step 5: Uncomment one of these based on what you want to run:

# For Terminal
# huggingface_qa_chatbot_terminal()

# For Streamlit Web
huggingface_qa_chatbot_streamlit()
