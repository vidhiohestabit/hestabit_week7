import sqlite3
from generator.sql_generator import generate_sql, validate_sql
from utils.schema_loader import load_schema

DB_PATH = "database/sample.db"


def execute_query(query):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    try:
        cur.execute(query)
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description]
        return cols, rows
    except Exception as e:
        return None, str(e)
    finally:
        conn.close()


def summarize_result(columns, rows):
    if not rows:
        return "No results found."

    result = ""
    for row in rows:
        result += ", ".join(str(x) for x in row) + "\n"

    return result


def run_sql_pipeline(question):
    # Step 1: Load schema
    schema = load_schema(DB_PATH)

    # Step 2: Generate SQL
    sql_query = generate_sql(question, schema)

    # Step 3: Validate SQL
    if not validate_sql(sql_query):
        return {
            "error": "Unsafe SQL detected",
            "sql": sql_query
        }

    # Step 4: Execute
    columns, rows = execute_query(sql_query)

    if columns is None:
        return {
            "error": rows,
            "sql": sql_query
        }

    # Step 5: Summarize
    summary = summarize_result(columns, rows)

    return {
        "sql": sql_query,
        "columns": columns,
        "rows": rows,
        "summary": summary
    }