from neo4j import GraphDatabase


url = 'neo4j://127.0.0.1:7687'
user_name = 'neo4j'
password = 'Sumanth-dl@orbit'

driver = GraphDatabase.driver(
    url, auth=(user_name, password)
)

with driver.session(database='neo4j') as session:
    result = session.run("RETURN 1 as number")
    print(result.single())

driver.close()