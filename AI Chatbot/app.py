import streamlit as st
from groq import Groq
from dotenv import load_dotenv
import os


# --------------------------------------------------
# 1. Load environment variables
# --------------------------------------------------

load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY")


# --------------------------------------------------
# 2. Check API key
# --------------------------------------------------

if not API_KEY:
    st.error("GROQ_API_KEY not found. Please add it to your .env file.")
    st.stop()


# --------------------------------------------------
# 3. Create Groq client
# --------------------------------------------------

client = Groq(api_key=API_KEY)


# --------------------------------------------------
# 4. Streamlit page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="AI Chatbot",
    page_icon="🤖",
    layout="centered"
)


# --------------------------------------------------
# 5. Application title
# --------------------------------------------------

st.title("🤖 AI Chatbot")
st.write("Ask me anything!")


# --------------------------------------------------
# 6. Initialize chat history
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful AI assistant. "
                "Give clear, accurate and beginner-friendly answers."
            )
        }
    ]


# --------------------------------------------------
# 7. Display previous messages
# --------------------------------------------------

for message in st.session_state.messages:

    if message["role"] == "system":
        continue

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# --------------------------------------------------
# 8. Get user input
# --------------------------------------------------

user_input = st.chat_input("Type your message...")


# --------------------------------------------------
# 9. Process user message
# --------------------------------------------------

if user_input:

    # Add user message to history
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input
        }
    )

    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)


    # --------------------------------------------------
    # 10. Generate AI response
    # --------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            try:

                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=st.session_state.messages,
                    temperature=0.7,
                    max_tokens=1024
                )

                assistant_response = response.choices[0].message.content

                st.markdown(assistant_response)


                # Save AI response
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": assistant_response
                    }
                )

            except Exception as e:

                st.error(
                    f"Something went wrong: {str(e)}"
                )