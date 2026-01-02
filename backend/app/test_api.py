import requests
import json

url = 'http://localhost:8000/api/v1/graph/fastapi/fastapi'
response = requests.get(url)
if response.status_code == 200:
    data = response.json()
    # Print nodes with cve_count
    for node in data.get('nodes', []):
        if 'cve_count' in node:
            print(f"Node {node['name']}: CVE count = {node['cve_count']}, risk_score = {node.get('risk_score')}")
else:
    print(f"Error: {response.status_code}")