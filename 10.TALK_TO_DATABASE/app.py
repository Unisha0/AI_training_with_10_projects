# app.py
import streamlit as st
from utils import initialize_database, text_to_sql, run_sql_query

# Initialize DB (only runs once)
initialize_database()

st.set_page_config("🧠 Talk with Your Database")
st.title("💬 Talk with your Database using Gemini")
st.markdown("Ask natural language questions about your student database!")

query = st.text_input("Type your question:", placeholder="e.g., Who are the students in the Engineering college")

if st.button("Submit"):
    if query:
        with st.spinner("Thinking..."):
            sql = text_to_sql(query)
            if sql:
                st.code(sql, language="sql")
                cols, rows = run_sql_query(sql)
                if cols and rows:
                    st.success("Here are the results:")
                    st.dataframe([dict(zip(cols, row)) for row in rows])
                else:
                    st.error("No results or query error.")
            else:
                st.error("Could not generate SQL query.")
    else:
        st.warning("Please enter a question.")