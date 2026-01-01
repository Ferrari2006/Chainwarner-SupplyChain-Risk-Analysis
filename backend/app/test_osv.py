import asyncio
import httpx

async def query_osv(package: str, ecosystem: str = 'PyPI') -> int:
    url = 'https://api.osv.dev/v1/query'
    body = {'package': {'name': package, 'ecosystem': ecosystem}}
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(url, json=body, timeout=5.0)
            if resp.status_code == 200:
                data = resp.json()
                return len(data.get('vulns', []))
    except:
        return 0
    return 0

print(asyncio.run(query_osv('django')))