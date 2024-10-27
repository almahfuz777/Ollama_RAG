import streamlit as st
from main import generate_ans
import logging

# Streamlit UI
st.set_page_config(
    page_icon='🔭',
    page_title="Physics ChatBot",
    layout="wide",
    # initial_sidebar_state="collapsed",
)
st.subheader("🤖 Q/A bot Physics")

# Sidebar for model selection and sessions
with st.sidebar:
    st.subheader("Chat Sessions", divider="gray") # Sidebar title

    # dropdown menu to select a model
    llm = st.selectbox(
        "Select LLM",
        options=["mistral","llama3.1", "llama3.2",],
        index=0  # Default selection
    )

    # Dropdown to select the embedding model/database
    embedding_model = st.selectbox(
        "Select Embedding Model",
        options=["all-MiniLM-L6-v2", "nomic-embed-text", ],
        index=0  # Default selection
    )
    
    # Button to clear the session (chat history and memory)
    if st.button("Clear Session"):
        st.session_state.clear()
        st.write("Session cleared.")
    
    # Blank Space to push the link to the bottom
    st.container(height=280, border=False)
        
    # Link to feedback form
    st.sidebar.markdown("[Share Feedback](https://forms.gle/qenvLzoynpdYuyit8)", unsafe_allow_html=True)

# Initialize session state for chat history 
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Initialize user_input
if "user_input" not in st.session_state:  
    st.session_state.user_input = ""

# Display the chat history
message_container = st.container(height=380, border=True)
with message_container:
    for message in st.session_state["chat_history"]:
        if "role" in message and "content" in message:
            avatar = "🤖" if message["role"]== "assistant" else "🤔"
            st.chat_message(message["role"], avatar=avatar).markdown(message["content"])  

# Input field for query
user_ques = st.text_input(
    label="Ask Bot", 
    value=st.session_state["user_input"], 
    placeholder="Throw your problems..."
)

# Submit button clicked
if st.button("Submit"):    
    if user_ques:
        try:
            # Append user ques to chat_history
            st.session_state.chat_history.append({
                "role": "user", 
                "content": user_ques
            })
            # Display the user question in the chat message container
            message_container.chat_message("user", avatar="🤔").markdown(user_ques)

            # Generate and display the assistant's response
            with message_container.chat_message("assistant", avatar="🤖"):
                with st.spinner("Fetching AI response..."):
                    final_ans = generate_ans(user_ques, llm, embedding_model)

                    # Display assistant response
                    st.markdown(final_ans)

                    # Append AI's response to chat history
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": final_ans
                    })
                    
                    st.session_state.user_input = ""    # Clear input after submission

        except Exception as e:
            st.error(f"An error occured: {str(e)}")
    else:
        st.write("Please enter a question!")


# logging after each response generation
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

def log_query(user_question, answer):
    logger.info(f"Question: {user_question}, Answer: {answer}")