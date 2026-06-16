from typing import Any, Dict, List

from models import Document


def dedup_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    deduplicated = []

    for item in items:
        key = (
            str(item.get("foodNm") or item.get("FOOD_NM_KR") or ""),
            str(item.get("foodCd") or item.get("ITEM_REPORT_NO") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        deduplicated.append(item)

    return deduplicated


def food_item_to_document(item: Dict[str, Any], index: int) -> Document:
    name = item.get("foodNm") or item.get("FOOD_NM_KR") or ""
    text = f"""
식품명: {name}
식품코드: {item.get("foodCd") or item.get("ITEM_REPORT_NO") or ""}
분류: {item.get("foodLv3Nm") or item.get("FOOD_CAT1_NM") or ""}
제조사/업체: {item.get("mfrNm") or item.get("MAKER_NM") or ""}

[영양]
에너지(kcal): {item.get("enerc", "")}
단백질(g): {item.get("prot", "")}
지방(g): {item.get("fatce", "")}
탄수화물(g): {item.get("chocdf", "")}
당류(g): {item.get("sugar", "")}
나트륨(mg): {item.get("nat", "")}

[원산지/출처]
원산지국명: {item.get("cooNm", "")}
출처명: {item.get("srcNm", "")}
""".strip()

    return Document(
        id=f"FOOD_{index:05d}",
        text=text,
        metadata=dict(item),
    )


def make_llm_context(docs: List[Document], max_chars: int = 4500) -> str:
    chunks = []
    total_length = 0

    for doc in docs:
        block = f"- 문서 {doc.id}\n{doc.text}\n"
        if total_length + len(block) > max_chars:
            break
        chunks.append(block)
        total_length += len(block)

    return "\n".join(chunks)
