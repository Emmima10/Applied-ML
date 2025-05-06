from transformers import pipeline
import streamlit as st

# Step 1: Your knowledge base
with open("chatbotkb.txt", "r", encoding="utf-8") as f:
    knowledge_base = f.read()
#knowledge_base = 
# """
# - The West region had the highest total sales at $725,458.
# - The most profitable product category was Technology.
# - The top-selling sub-category by revenue was Phones.
# - Discounts above 30% were associated with an average profit of $-107.21.
# -Cluster 0.0: Sales: low , Profit: low , Discount: high , Quantity: high . 
# This cluster has low engagement and is unprofitable. Focus on re-engaging these customers with promotions or incentives.
# -Cluster 1.0: 
# Sales: high , Profit: high , Discount: low , Quantity: high . 
# This cluster represents high-value customers who generate substantial revenue and profit. Consider offering premium services or upselling opportunities.
# -Cluster 2.0: 
# Sales: low , Profit: low , Discount: low , Quantity: low . 
# This cluster has low engagement and is unprofitable. Focus on re-engaging these customers with promotions or incentives

# """

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
    # 💡 Suggested Questions
    st.markdown("### 💡 Try asking:")
    st.markdown("- What is Cluster 0?")
    st.markdown("- Who are our best customers?")
    st.markdown("- Which region has the highest sales?")
    st.markdown("- How did Prophet perform?")
    st.markdown("- Which product category is most profitable?")
    st.markdown("- How many days are forecasted with sales below $10,000?")
    
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
