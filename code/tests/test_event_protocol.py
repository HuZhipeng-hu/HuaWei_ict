from scripts.validate_event_protocol import estimate_rates_from_frames


def test_protocol_rate_estimation_matches_500hz_emg_and_50hz_imu():
    frames = []
    for index in range(10):
        frames.append(
            {
                "timestamp": index * 20,
                "emg": [[0] * 8 for _ in range(10)],
            }
        )

    report = estimate_rates_from_frames(frames)

    assert report["frame_count"] == 10
    assert report["emg_pack_count"] == 100
    assert abs(report["timestamp_frame_rate_hz"] - 50.0) < 1e-6
    assert abs(report["timestamp_emg_rate_hz"] - 500.0) < 1e-6
