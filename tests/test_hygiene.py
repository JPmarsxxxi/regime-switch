import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_every_deviation_is_referenced_in_code_and_every_reference_exists():
    text = (ROOT / "DEVIATIONS.md").read_text()
    anchors = set(re.findall(r'<a id="([A-Z]\d+)"></a>', text))
    code = "\n".join(p.read_text() for p in list((ROOT / "regime_switch").glob("*.py")) + list((ROOT / "scripts").glob("*.py")))
    refs = set(re.findall(r"DEVIATIONS\.md#([A-Z]\d+)", code))
    ranges = re.findall(r"DEVIATIONS\.md#([A-Z])(\d+)-[A-Z]?(\d+)", code)
    for letter, a, b in ranges:
        refs |= {f"{letter}{i}" for i in range(int(a), int(b) + 1)}
    assert refs <= anchors, f"references without an entry: {sorted(refs - anchors)}"
    assert anchors <= refs, f"entries never referenced in code: {sorted(anchors - refs)}"
