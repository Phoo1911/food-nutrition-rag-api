# api_client.py
from typing import Any, Dict, List, Optional
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config import API_DEFS, CONNECT_TIMEOUT, READ_TIMEOUT, TOTAL_RETRIES

def _session() -> requests.Session:
    s = requests.Session()

    # ✅ HPC/서버 환경에서 proxy 환경변수 때문에 막히는 경우가 많음
    s.trust_env = False

    retry = Retry(
        total=TOTAL_RETRIES,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    s.headers.update({"User-Agent": "Mozilla/5.0 (FoodRAG/1.0)"})
    return s

SESSION = _session()

def compact_params(params: Dict[str, Any]) -> Dict[str, Any]:
    keep = {}
    for k, v in params.items():
        if v is None:
            continue
        if isinstance(v, str) and v.strip() == "":
            continue
        keep[k] = v
    return keep

def call_public_api(
    api_name: str,
    service_key: str,
    keyword: str,
    pageNo: int = 1,
    numOfRows: int = 100,
    resp_type: str = "json",
    extra_params: Optional[Dict[str, Any]] = None,
) -> Any:
    if api_name not in API_DEFS:
        raise ValueError(f"Unknown api: {api_name}")

    api = API_DEFS[api_name]
    url = api["url"]

    params = dict(api["default_params"])
    params["serviceKey"] = service_key
    params["pageNo"] = str(pageNo)
    params["numOfRows"] = str(numOfRows)
    params["type"] = resp_type

    kf = api["keyword_field"]
    params[kf] = keyword

    if extra_params:
        params.update(extra_params)

    send_params = compact_params(params)

    r = SESSION.get(url, params=send_params, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT))
    r.raise_for_status()

    return r.json() if resp_type.lower() == "json" else r.text

def extract_items(data: Any, resp_type: str = "json") -> List[Dict[str, Any]]:
    if resp_type != "json" or not isinstance(data, dict):
        return []
    body = data.get("response", {}).get("body", {})
    items = body.get("items", {})
    if isinstance(items, dict) and "item" in items:
        it = items["item"]
        return it if isinstance(it, list) else [it]
    if isinstance(items, list):
        return items
    return []
