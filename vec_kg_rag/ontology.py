from __future__ import annotations

ENTITY_TYPES = {
    "College",
    "Department",
    "Program",
    "Faculty",
    "Lab",
    "Media",
    "Resource",
    "Alumni",
    "IndustryPartner",
    "Internship",
    "CareerPath",
}

RELATION_TYPES = {
    "HAS_DEPARTMENT",
    "HAS_PROGRAM",
    "HAS_FACULTY",
    "HAS_LAB",
    "HAS_RESOURCE",
    "HAS_MEDIA",
    "COOPERATES_WITH",
    "OFFERS_INTERNSHIP",
    "LEADS_TO_CAREER",
    "BELONGS_TO",
    "ADVISED_BY",
}


def normalize_entity_type(value: str) -> str:
    value = (value or "").strip()
    if value in ENTITY_TYPES:
        return value
    return "Resource"


def normalize_relation_type(value: str) -> str:
    value = (value or "").strip().upper()
    if value in RELATION_TYPES:
        return value
    if value.startswith("OTHER_") and len(value) > 6:
        return value
    return "OTHER_RELATED_TO"
