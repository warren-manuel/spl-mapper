import requests
from lxml import etree
import re

DAILYMED_BASE = "https://dailymed.nlm.nih.gov/dailymed/services/v2"
CONTRA_Loinc = "34070-3"
ADVERSE_Loinc = "34084-4"
# https://dailymed.nlm.nih.gov/dailymed/services/v2/spls/{SETID}.xml

class DailyMedError(Exception):
    pass


_BLOCK_TAGS = {
    "caption",
    "footnote",
    "item",
    "list",
    "paragraph",
    "renderMultiMedia",
    "table",
    "tbody",
    "td",
    "tfoot",
    "th",
    "thead",
    "tr",
}

_DOSAGE_FORM_BOUNDARY = re.compile(
    r"(?<=[A-Za-z])(?=(?:capsule|capsules|cream|drop|drops|gel|inhalation|injection|"
    r"injectable|lotion|ointment|patch|powder|solution|spray|suspension|syrup|tablet|"
    r"tablets)\b)"
)


def _local_name(el: etree._Element) -> str:
    return etree.QName(el).localname if isinstance(el.tag, str) else ""


def _append_text(parts: list[str], value: str | None) -> None:
    if value:
        parts.append(value)


def _walk_narrative(el: etree._Element, parts: list[str]) -> None:
    tag = _local_name(el)
    is_block = tag in _BLOCK_TAGS

    if is_block and parts and not parts[-1].endswith("\n"):
        parts.append("\n")

    if tag == "br":
        parts.append("\n")
        return

    _append_text(parts, el.text)

    for child in el:
        _walk_narrative(child, parts)
        _append_text(parts, child.tail)

    if is_block and parts and not parts[-1].endswith("\n"):
        parts.append("\n")


def _normalize_narrative_text(text: str) -> str:
    text = text.replace("\xa0", " ").replace("‚Ä¢", "•").replace("â€¢", "•")
    text = _DOSAGE_FORM_BOUNDARY.sub(" ", text)
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"([\[\(])\n+", r"\1", text)
    text = re.sub(r"\n+([\]\)])", r"\1", text)
    text = re.sub(r" +([,.;:?!\])])", r"\1", text)
    text = re.sub(r"([\[\(]) +", r"\1", text)
    text = re.sub(r"(?<!\n)•\s*", "\n• ", text)
    text = re.sub(r"\n•\s+", "\n• ", text)
    return text.strip()

def fetch_spl_xml_by_setid(setid: str, timeout: int = 30) -> bytes:
    """
    Download the SPL XML for a given DailyMed SETID.
    """
    url = f"{DAILYMED_BASE}/spls/{setid}.xml"  # returns the full SPL document (v2)
    resp = requests.get(url, timeout=timeout)
    if resp.status_code != 200 or not resp.content.strip():
        raise DailyMedError(f"Failed to fetch SPL XML for setid={setid}: {resp.status_code}")
    return resp.content

def parse_xml(xml_bytes: bytes) -> etree._Element:
    """
    Parse bytes into an lxml Element tree with recovery on.
    """
    parser = etree.XMLParser(remove_blank_text=True, recover=True, huge_tree=True)
    return etree.fromstring(xml_bytes, parser=parser)

def get_default_ns(root: etree._Element) -> str:
    """
    SPLs typically use the default namespace urn:hl7-org:v3. Detect it from the root.
    """
    return root.nsmap.get(None, "urn:hl7-org:v3")

def find_section_by_loinc(root: etree._Element, loinc_code: str) -> etree._Element | None:
    """
    Return the <section> element whose <code/@code> == loinc_code.
    Works regardless of the section @ID (e.g., s10).
    """
    ns = {"hl7": get_default_ns(root)}
    # Find the section by its LOINC code anywhere in the document
    xpath = ".//hl7:component[hl7:section/hl7:code[@code=$code]]/hl7:section"
    results = root.xpath(xpath, namespaces=ns, code=loinc_code)
    return results[0] if results else None


def get_product_name(root: etree._Element) -> str | None:
    """
    Extract the SPL product name from the document header.

    Preference order:
    1. SPL product data elements section (LOINC 48780-1)
    2. document/title
    3. manufacturedProduct/manufacturedMedicine/name
    4. manufacturedProduct/name
    """
    ns = {"hl7": get_default_ns(root)}
    candidates = [
        (
            "normalize-space(.//hl7:section[hl7:code[@code='48780-1']]"
            "/hl7:subject/hl7:manufacturedProduct/hl7:manufacturedProduct/hl7:name[1])"
        ),
        "normalize-space(/hl7:document/hl7:title)",
        "normalize-space(.//hl7:manufacturedProduct//hl7:manufacturedMedicine/hl7:name[1])",
        "normalize-space(.//hl7:manufacturedProduct//hl7:name[1])",
    ]

    for xpath in candidates:
        value = root.xpath(xpath, namespaces=ns)
        if isinstance(value, str) and value:
            return value
    return None

def section_text(section_el: etree._Element) -> str:
    """
    Extract human-readable text from the section's <text> narrative block.
    Preserves paragraph/list breaks reasonably well.
    """
    if section_el is None:
        return ""
    ns = {"hl7": section_el.nsmap.get(None, "urn:hl7-org:v3")}
    text_el = section_el.find(".//hl7:text", namespaces=ns)
    if text_el is None:
        return ""
    parts: list[str] = []
    _walk_narrative(text_el, parts)
    return _normalize_narrative_text("".join(parts))

def extract_section(setid: str, loinc_code: list[str]) -> dict:
    """
    Fetch SPL by setid and extract a specific section.
    Returns:
        {
          "setid": str,
          "found": bool,
          "section_xml": str | None,
          "section_text": str | None
        }
    """
    xml_bytes = fetch_spl_xml_by_setid(setid)
    root = parse_xml(xml_bytes)

    sections = {}
    for code in loinc_code:
        section = find_section_by_loinc(root, code)
        if section is not None:
            sections[code] = {
                "section_xml": etree.tostring(section, pretty_print=True, encoding="unicode"),
                "section_text": section_text(section)
            }

    return {
        "setid": setid,
        "product_name": get_product_name(root),
        "found": bool(sections),
        "sections": sections
    }