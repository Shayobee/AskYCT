import streamlit as st
import funcs as fc

st.markdown(f"""
<img style="border: 2px solid powderblue" src="data:image/jpeg;base64,{fc.open_picture("chat.webp")}" width="20%"><br>""",
            unsafe_allow_html=True)
st.title("Chat YCT")

st.chat_message('assistant').markdown("""
Hello, I'm ChatYCT! 🎓
I help with admission queries for Yaba college of Technology. Here's what I can do:
- JAMB Subjects: Right subjects for your program.
- O'level Requirements: Compulsory & elective subjects.
- Cut-Off Marks: Minimum scores for admission.
- Program Availability: Full-time, part-time options.
- Program Types: HND, OND, or B.Ed.
- Admission Requirements: Criteria for your preferred course of study.

Ask me anything about admissions—I'm here to help! 😊
What would you like to know today?""")

if 'messages' not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])


chat_prompt = st.chat_input("Prompt")
if chat_prompt:
    st.chat_message('user').markdown(chat_prompt)
    user_raw_input = fc.preprocess_input(chat_prompt)

    # bot_reply = fc.queries(chat_prompt)
    bot_reply = fc.queries(user_raw_input)
    """"""

    # print("BOT: ", bot_reply)
    st.session_state.messages.append({'role': 'user', 'content': chat_prompt})

    st.chat_message('assistant').markdown(bot_reply)
    st.session_state.messages.append({'role': 'assistant', 'content': bot_reply})
