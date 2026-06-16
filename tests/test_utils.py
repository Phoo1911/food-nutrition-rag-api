from utils import dedup_items, food_item_to_document, make_llm_context


def test_dedup_items_removes_duplicate_food_entries() -> None:
    items = [
        {"foodNm": "신라면", "foodCd": "100"},
        {"foodNm": "신라면", "foodCd": "100"},
        {"foodNm": "진라면", "foodCd": "200"},
    ]

    deduplicated = dedup_items(items)

    assert len(deduplicated) == 2


def test_food_item_to_document_builds_readable_text() -> None:
    document = food_item_to_document(
        {
            "foodNm": "신라면",
            "foodCd": "100",
            "enerc": "500",
            "prot": "10",
            "fatce": "15",
            "chocdf": "70",
            "sugar": "5",
            "nat": "1800",
        },
        0,
    )

    assert document.id == "FOOD_00000"
    assert "신라면" in document.text
    assert "에너지(kcal): 500" in document.text


def test_make_llm_context_respects_character_limit() -> None:
    docs = [
        food_item_to_document({"foodNm": "A", "foodCd": "1"}, 0),
        food_item_to_document({"foodNm": "B", "foodCd": "2"}, 1),
    ]

    context = make_llm_context(docs, max_chars=60)

    assert len(context) <= 60
