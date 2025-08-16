# utils.py
import sqlite3
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

# Gemini API Key
GEMINI_API_KEY = "AIzaSyDWGKzWnClG8B5t3FSwfy26i6prqQbKLmw"

# Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=GEMINI_API_KEY,
    temperature=0.3,
    max_tokens=500,
)

def initialize_database():
    if os.path.exists("student_management.db"):
        return

    conn = sqlite3.connect("student_management.db")
    cursor = conn.cursor()

    # Create Faculties Table
    cursor.execute('''
        CREATE TABLE faculties (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            building TEXT
        )
    ''')

    # Create Courses Table
    cursor.execute('''
        CREATE TABLE courses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            course_name TEXT NOT NULL,
            credit_hours INTEGER,
            faculty_id INTEGER,
            FOREIGN KEY(faculty_id) REFERENCES faculties(id)
        )
    ''')

    # Create Students Table
    cursor.execute('''
        CREATE TABLE students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            first_name TEXT NOT NULL,
            last_name TEXT NOT NULL,
            faculty_id INTEGER,
            enrollment_year INTEGER,
            gpa REAL,
            FOREIGN KEY(faculty_id) REFERENCES faculties(id)
        )
    ''')

    # Insert Faculties
    cursor.executemany('INSERT INTO faculties (name, building) VALUES (?, ?)', [
        ("Computer Engineering", "Block A"),
        ("Electronics Engineering", "Block B"),
        ("Civil Engineering", "Block C"),
        ("Architecture", "Block D"),
        ("Mechanical Engineering", "Block E")
    ])

    # Insert Courses
    cursor.executemany('INSERT INTO courses (course_name, credit_hours, faculty_id) VALUES (?, ?, ?)', [
        ("Data Structures", 3, 1),
        ("Digital Logic", 3, 2),
        ("Structural Analysis", 4, 3),
        ("Design Studio", 5, 4),
        ("Thermodynamics", 3, 5)
    ])

    # Insert Students
    cursor.executemany('INSERT INTO students (first_name, last_name, faculty_id, enrollment_year, gpa) VALUES (?, ?, ?, ?, ?)', [
        ("Yunisha", "Chaulagain", 1, 2022, 3.8),
        ("Nayana", "Bhatta", 2, 2021, 3.5),
        ("Ashmita", "Thapa", 3, 2023, 3.9),
        ("Jeevika", "Subedi", 4, 2022, 3.7)
    ])

    conn.commit()
    conn.close()

def call_gemini_with_langchain(prompt):
    message = HumanMessage(content=prompt)
    try:
        response = llm.invoke([message])
        return response.content.strip().replace("```sql", "").replace("```", "").strip()
    except Exception as e:
        return f"Error: {e}"

def text_to_sql(question):
    table_info = """
    The database contains the following tables:

    1. students table:
    - id (INTEGER PRIMARY KEY)
    - first_name (TEXT)
    - last_name (TEXT)
    - faculty_id (INTEGER)
    - enrollment_year (INTEGER)
    - gpa (REAL)

    2. faculties table:
    - id (INTEGER PRIMARY KEY)
    - name (TEXT)
    - building (TEXT)

    3. courses table:
    - id (INTEGER PRIMARY KEY)
    - course_name (TEXT)
    - credit_hours (INTEGER)
    - faculty_id (INTEGER)
    """

    prompt = f"""
    You are an expert that translates natural language into SQL queries.
    Based on the database schema below, write a single valid SQLite query.
    Return only the SQL query, no explanation.

    ---
    SCHEMA:
    {table_info}
    ---
    QUESTION: {question}
    ---
    SQL QUERY:
    """

    return call_gemini_with_langchain(prompt)

def run_sql_query(query):
    try:
        conn = sqlite3.connect("student_management.db")
        cursor = conn.cursor()
        cursor.execute(query)
        results = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        conn.close()
        return columns, results
    except sqlite3.Error as e:
        return None, f"SQL Error: {e}"