import textwrap

from VaxMapper.src.utils.dailymed import (
    find_section_by_loinc,
    get_product_name,
    parse_xml,
    section_text,
)


def test_section_text_preserves_list_breaks_and_inline_text():
    xml = textwrap.dedent(
        """\
        <document xmlns="urn:hl7-org:v3">
          <component>
            <section ID="Section_4">
              <code code="34070-3" />
              <text>
                <paragraph>
                  <content styleCode="xmChange">•&#160;Eszopiclonetablets are contraindicated in patients who have experienced complex sleep behaviors after taking eszopiclone tablets [
                    <content styleCode="italics">see Warnings and Precautions (
                      <linkHtml href="#Section_5.8">5.1</linkHtml>)
                    </content>].
                    <br/>•&#160;Eszopiclone tablets are contraindicated in patients with known hypersensitivity to eszopiclone. Hypersensitivity reactions include anaphylaxis and angioedema [
                    <content styleCode="italics">see Warnings and Precautions (
                      <linkHtml href="#Section_5.2">5.3</linkHtml>)
                    </content>].
                  </content>
                </paragraph>
              </text>
            </section>
          </component>
        </document>
        """
    ).encode("utf-8")

    root = parse_xml(xml)
    section = find_section_by_loinc(root, "34070-3")

    assert section is not None
    assert section_text(section) == textwrap.dedent(
        """\
        • Eszopiclone tablets are contraindicated in patients who have experienced complex sleep behaviors after taking eszopiclone tablets [see Warnings and Precautions (5.1)].
        • Eszopiclone tablets are contraindicated in patients with known hypersensitivity to eszopiclone. Hypersensitivity reactions include anaphylaxis and angioedema [see Warnings and Precautions (5.3)].
        """
    ).strip()


def test_get_product_name_prefers_product_data_section():
    xml = textwrap.dedent(
        """\
        <document xmlns="urn:hl7-org:v3">
          <title>Fallback Title</title>
          <component>
            <structuredBody>
              <component>
                <section>
                  <code code="48780-1" />
                  <subject>
                    <manufacturedProduct>
                      <manufacturedProduct>
                        <name>Eszopiclone</name>
                      </manufacturedProduct>
                    </manufacturedProduct>
                  </subject>
                </section>
              </component>
            </structuredBody>
          </component>
        </document>
        """
    ).encode("utf-8")

    root = parse_xml(xml)

    assert get_product_name(root) == "Eszopiclone"
