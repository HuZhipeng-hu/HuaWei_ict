import numpy as np

from training.data.split_strategy import build_manifest


def test_manifest_v2_group_no_leak():
    labels = np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)
    source_ids = np.array(
        [
            "RELAX/a.csv",
            "RELAX/a.csv",
            "RELAX/b.csv",
            "FIST/c.csv",
            "FIST/c.csv",
            "FIST/d.csv",
        ],
        dtype=object,
    )
    metadata = [
        {"recording_id": "a", "session_id": "s1"},
        {"recording_id": "a", "session_id": "s1"},
        {"recording_id": "b", "session_id": "s2"},
        {"recording_id": "c", "session_id": "s1"},
        {"recording_id": "c", "session_id": "s1"},
        {"recording_id": "d", "session_id": "s2"},
    ]
    manifest = build_manifest(
        labels,
        source_ids,
        seed=7,
        split_mode="grouped_file",
        val_ratio=0.2,
        test_ratio=0.2,
        num_classes=2,
        class_names=["RELAX", "FIST"],
        manifest_strategy="v2",
        source_metadata=metadata,
    )

    train = set(manifest.group_keys_train)
    val = set(manifest.group_keys_val)
    test = set(manifest.group_keys_test)
    assert not (train & val)
    assert not (train & test)
    assert not (val & test)


def test_manifest_v2_no_group_leak_when_same_recording_contains_multiple_classes():
    labels = np.array([0, 1, 0, 1, 0, 1], dtype=np.int32)
    source_ids = np.array(
        [
            "user_id=S1|session_id=S1_A1|recording_id=R1|x=0",
            "user_id=S1|session_id=S1_A1|recording_id=R1|x=1",
            "user_id=S2|session_id=S2_A1|recording_id=R2|x=0",
            "user_id=S2|session_id=S2_A1|recording_id=R2|x=1",
            "user_id=S3|session_id=S3_A1|recording_id=R3|x=0",
            "user_id=S3|session_id=S3_A1|recording_id=R3|x=1",
        ],
        dtype=object,
    )
    manifest = build_manifest(
        labels,
        source_ids,
        seed=11,
        split_mode="grouped_file",
        val_ratio=0.2,
        test_ratio=0.2,
        num_classes=2,
        class_names=["C0", "C1"],
        manifest_strategy="v2",
        source_metadata=None,
    )

    train = set(manifest.group_keys_train)
    val = set(manifest.group_keys_val)
    test = set(manifest.group_keys_test)
    assert not (train & val)
    assert not (train & test)
    assert not (val & test)
