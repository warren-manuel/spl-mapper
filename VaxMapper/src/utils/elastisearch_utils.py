# Contents of /rxnorm-term-getter/rxnorm-term-getter/src/utils/elasticsearch_utils.py
import os
import subprocess
import time
import requests
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional

import pandas as pd
import requests
from elasticsearch import Elasticsearch
from elasticsearch.helpers import streaming_bulk

# --------- Defaults (can be overridden via env or arguments) --------- #

DEFAULT_ES_HOME = Path(
    os.environ.get(
        "ES_HOME",
        "/data/wmanuel3/VaxMapperRepo/esdata/elasticsearch-9.0.0"
    )
)
DEFAULT_ES_PORT = int(os.environ.get("ES_PORT", 9200))
DEFAULT_ES_HOST = os.environ.get("ES_HOST", "http://localhost")
DEFAULT_JAVA_OPTS = os.environ.get("ES_JAVA_OPTS", "-Xms2g -Xmx2g")

# --------- Elasticsearch Management Utilities --------- #

# ES_HOME = Path("/data/wmanuel3/VaxMapperRepo/esdata/elasticsearch-9.0.0")
# ES_PORT = 9200

def run_elasticsearch(
    es_home: Path = DEFAULT_ES_HOME,
    port: int = DEFAULT_ES_PORT,
    java_opts: str = DEFAULT_JAVA_OPTS,
    timeout_seconds: int = 120,
) -> subprocess.Popen:
    
    es_executable = es_home / "bin/elasticsearch"
    env = os.environ.copy()
    env.update({
        "ES_JAVA_OPTS": java_opts,
        "discovery.type": "single-node",
        "xpack.security.enabled": "false"
    })
    print(f"Starting Elasticsearch from {es_executable}: on port {port}...")
    process = subprocess.Popen(
        [str(es_executable)], 
        env=env, 
        cwd=es_home,
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE
    )

    base_url = f"http://localhost:{port}"
    for _ in range(timeout_seconds // 2):
        try:
            r = requests.get(base_url, timeout=2)
            if r.status_code == 200:
                print(f"Elasticsearch is running: {r.json().get('version', {}).get('number')}")
                return process
        except (requests.ConnectionError, requests.Timeout):
            time.sleep(2)
    print("Elasticsearch did not start within timeout.")
    return process

def stop_elasticsearch(process: subprocess.Popen):
    if process is None:
        return
    process.terminate()
    time.sleep(5)
    if process.poll() is None:
        print("Sending SIGKILL...")
        process.kill()

# --------- Client creation --------- #

def get_es_client(
    host: str = DEFAULT_ES_HOST,
    port: int = DEFAULT_ES_PORT,
    **kwargs,
) -> Elasticsearch:
    """
    Create an Elasticsearch client. You can override host/port or pass additional
    kwargs (e.g., basic_auth, ssl_context, etc.).
    """
    url = f"{host}:{port}" if not host.endswith(str(port)) else host
    return Elasticsearch(url, **kwargs)

# --------- Index management --------- #

def create_index(
        es: Elasticsearch, 
        index_name: str,
        settings: Optional[Dict] = None,
        delete_if_exists: bool = False,
    ):
    if es.indices.exists(index=index_name):
        if delete_if_exists:
            print(f"Deleting existing index: {index_name}")
            es.indices.delete(index=index_name)
        else:
            print(f"Index '{index_name}' already exists; skipping creation.")
            return

    if settings is None:
        settings = {}  # ES will use defaults
    print(f"Creating index '{index_name}'...")
    es.indices.create(index=index_name, body=settings, ignore=400)



# --------- Bulk indexing from DataFrame --------- #

def doc_actions(
        df: pd.DataFrame, 
        index_name: str, 
        # id_field: str = "vo_id", 
        id_col: Optional[str] = None,
        field_map: Optional[Dict[str, str]] = None,
        doc_transform: Optional[Callable[[Dict, pd.Series], Dict]] = None,
        # text_field: str = "label"
        ) -> Iterable[Dict]:
    df = df.fillna("")

    if field_map is None:
        #every column becomes a field with the same name
        field_map = {col: col for col in df.columns}

    for idx, row in df.iterrows():
        # yield {
        #     "_op_type": "index",
        #     "_index": index_name,
        #     "_id": idx,
        #     id_field: row[id_field],
        #     text_field: row[text_field]
        # }
        doc = {es_field: row[df_col] for df_col, es_field in field_map.items()}

        if doc_transform is not None:
            doc = doc_transform(doc, row)

        action = {
            "_op_type": "index",
            "_index": index_name,
            "_source": doc,
        }

        if id_col is not None and id_col in row:
            action["_id"] = row[id_col]
        else:
            action["_id"] = idx

        yield action


def bulk_index(
        es: Elasticsearch, 
        df: pd.DataFrame, 
        index_name: str,
        id_col: Optional[str] = None,
        field_map: Optional[Dict[str, str]] = None,
        doc_transform: Optional[Callable[[Dict, pd.Series], Dict]] = None,
        chunk_size: int = 1000, 
        # id_field: str = "vo_id", 
        # text_field: str = "label"
        ):
    
    actions = doc_actions(
        df=df, 
        index_name=index_name,
        id_col=id_col,
        field_map=field_map,
        doc_transform=doc_transform
    )

    for ok, result in streaming_bulk(
        es, actions, chunk_size=chunk_size, raise_on_error=False
    ):
        if not ok:
            print("Failed to index:", result)