import time, requests, docker
from pathlib import Path

ES_IMAGE = "docker.elastic.co/elasticsearch/elasticsearch:8.18.0"
ES_PORT  = 9200
DATA_DIR = Path("/data/wmanuel3/VaxMapperRepo/esdata")
DATA_DIR.mkdir(parents=True, exist_ok=True)

client = docker.from_env()
client.images.pull(ES_IMAGE)

container = client.containers.run(
    ES_IMAGE,
    name="es-bm25",
    detach=True,
    ports={f"{ES_PORT}/tcp": ES_PORT},
    environment={
        "xpack.security.enabled": "false",
        "discovery.type": "single-node",
        "ES_JAVA_OPTS": "-Xms2g -Xmx2g",
    },
    volumes={str(DATA_DIR): {"bind": "/usr/share/elasticsearch/data", "mode": "rw"}},
    restart_policy={"Name": "unless-stopped"},
)

print("⏳ waiting for ES to boot …")
for _ in range(60):
    try:
        r = requests.get(f"http://localhost:{ES_PORT}")
        if r.ok:
            print("✅ Elasticsearch is up:", r.json()["version"]["number"])
            break
    except requests.ConnectionError:
        pass
    time.sleep(2)
else:
    raise RuntimeError("Elasticsearch did not become ready in 120 s")
