from groq import Groq

# Initialize client (uses GROQ_API_KEY from environment)
client = Groq()

def generate_sql(question: str, schema: str) -> str:
    prompt = f"""
You are an expert SQL generator.

Database Schema:
{schema}

User Question:
{question}

Rules:
- Only return SQL query
- No explanation
- Use correct table/column names
- Only SELECT queries
- Add LIMIT if user asks for top results
- Use aggregation (SUM, COUNT) when needed
- Use GROUP BY when aggregation is used

SQL:
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    query = response.choices[0].message.content.strip()

    # Clean markdown if model returns ```sql
    query = query.replace("```sql", "").replace("```", "").strip()

    return query


def validate_sql(query: str) -> bool:
    dangerous_keywords = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER"]

    for keyword in dangerous_keywords:
        if keyword in query.upper():
            return False

    return query.strip().lower().startswith("select")