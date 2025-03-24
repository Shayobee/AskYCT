import streamlit as st

homepage = st.Page("webpages/home.py", title="About Project", icon=":material/all_inclusive:")
chatbot = st.Page("webpages/chatbot.py", title="Chat YCT", icon=":material/chat:")
# team_page = st.Page("webpages/the_team.py", title="The team", icon=":material/groups:")

pg = st.navigation([chatbot, homepage])

pg.run()
