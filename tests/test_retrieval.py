from retrieval import RetrievalIndex, normalize_text


def test_normalize_text_folds_case_and_yo():
    assert normalize_text("ЁлКа") == "елка"


def test_normalize_text_collapses_whitespace():
    assert normalize_text("  Ремонт   телевизора\n") == "ремонт телевизора"


def test_retrieval_prefers_relevant_local_item():
    items = [
        {
            "item_id": "0000000000000001",
            "item_title_raw": "Ремонт телевизоров",
            "item_description_raw": "Починим телевизор дома",
            "item_infm_params_text": "",
            "item_location_id": 7,
        },
        {
            "item_id": "0000000000000002",
            "item_title_raw": "Ремонт телевизоров",
            "item_description_raw": "Починим телевизор дома",
            "item_infm_params_text": "",
            "item_location_id": 8,
        },
        {
            "item_id": "0000000000000003",
            "item_title_raw": "Массаж",
            "item_description_raw": "Массаж спины",
            "item_infm_params_text": "",
            "item_location_id": 7,
        },
    ]
    index = RetrievalIndex(items)
    result = index.search("ремонт телевизора", location_id=7, limit=2)
    assert result == ["0000000000000001", "0000000000000002"]


def test_retrieval_caps_output_and_deduplicates_items():
    items = [
        {
            "item_id": "0000000000000001",
            "item_title_raw": "Уборка квартиры",
            "item_description_raw": "",
            "item_infm_params_text": "",
            "item_location_id": 1,
        },
        {
            "item_id": "0000000000000001",
            "item_title_raw": "Уборка квартиры",
            "item_description_raw": "",
            "item_infm_params_text": "",
            "item_location_id": 1,
        },
    ]
    assert RetrievalIndex(items).search("уборка", location_id=1) == ["0000000000000001"]
