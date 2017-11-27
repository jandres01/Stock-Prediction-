import sqlite3
from sqlite3 import Error
 
def create_connection(db_file):
    """ create a database connection to the SQLite database
        specified by the db_file
    :return: Connection object or None
    """
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except Error as e:
        print(e)
    return None
 
def select_all_tasks(conn):
    cur = conn.cursor()
    cur.execute("SELECT * FROM crsp LIMIT 10")
    rows = cur.fetchall()
    for row in rows:
        print(row)
 
def select_task_by_priority(conn, priority):
    cur = conn.cursor()
    cur.execute("SELECT * FROM crsp")
    rows = cur.fetchall()
    for row in rows:
        print(row)
 
def main():
    database = '/home/jandres/data/financial-data.sqlite'
    conn = create_connection(database)
    with conn:
        print("Query all tasks")
        select_all_tasks(conn)
 
if __name__ == '__main__':
    main()

