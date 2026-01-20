@echo off
echo Activating Neo4j environment...
call conda activate neo4j_drivers_poc

echo Loading sample data into student-placement database...
python load_sample_data.py

echo Starting Two-Agent Agentic Neo4j System...
python langchain_neo4j_agent.py

pause