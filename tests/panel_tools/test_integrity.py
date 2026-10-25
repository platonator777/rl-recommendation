import json

from tlab.panel_tools.integrity import hash_tree, verify_protected_tree


def test_integrity_hashes_are_read_only_and_detect_changed_files(tmp_path):
    protected = tmp_path / "submission"
    protected.mkdir()
    original = protected / "payload.txt"
    original.write_text("fixed", encoding="utf-8")
    before = hash_tree(protected)
    manifest = tmp_path / "before.json"
    manifest.write_text(json.dumps(before), encoding="utf-8")
    assert verify_protected_tree(protected, manifest)["status"] == "unchanged"
    original.write_text("changed", encoding="utf-8")
    result = verify_protected_tree(protected, manifest)
    assert result["status"] == "MISMATCH"
    assert result["changed"] == ["payload.txt"]
