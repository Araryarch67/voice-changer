#![cfg(all(feature = "audio", target_os = "linux"))]
use std::{
    sync::atomic::Ordering,
    time::{Duration, Instant},
};
use voice_changer::{
    audio::{AudioOptions, AudioSession},
    config::Settings,
};

#[test]
#[ignore = "requires ALSA_CONFIG_PATH pointing to a null-only ALSA configuration"]
fn audio_worker_starts_processes_and_stops_with_null_devices() {
    assert_eq!(std::env::var("VC_TEST_NULL_AUDIO").as_deref(), Ok("1"), "Set VC_TEST_NULL_AUDIO=1 with a null-only ALSA configuration; never test against a real microphone");
    let record_dir = std::env::temp_dir().join(format!("vc-rec-test-{}", std::process::id()));
    let session = AudioSession::start(AudioOptions {
        input: None,
        output: None,
        manifest: None,
        settings: Settings::default(),
        record_dir: record_dir.clone(),
        virtual_mic: false,
        monitor: None,
    })
    .unwrap();
    let start = Instant::now();
    while !session.metrics.ready.load(Ordering::Acquire)
        && session.metrics.running.load(Ordering::Acquire)
        && start.elapsed() < Duration::from_secs(5)
    {
        std::thread::sleep(Duration::from_millis(10));
    }
    assert!(
        session.metrics.ready.load(Ordering::Acquire),
        "{:?}",
        session.metrics.error.lock().unwrap()
    );
    // REC: long enough for at least one 320 ms block to be written.
    session.set_recording(true);
    std::thread::sleep(Duration::from_millis(900));
    session.stop();
    let start = Instant::now();
    while session.metrics.running.load(Ordering::Acquire)
        && start.elapsed() < Duration::from_secs(5)
    {
        std::thread::sleep(Duration::from_millis(10));
    }
    assert!(!session.metrics.running.load(Ordering::Acquire));
    assert!(session.metrics.error.lock().unwrap().is_none());
    assert!(session.metrics.input_rate.load(Ordering::Relaxed) > 0);
    let recorded = session.metrics.recorded.lock().unwrap().take();
    assert!(
        recorded
            .as_deref()
            .is_some_and(|m| m.starts_with("Recording saved")),
        "{recorded:?}"
    );
    let files: Vec<_> = std::fs::read_dir(&record_dir)
        .unwrap()
        .flatten()
        .map(|entry| entry.path())
        .collect();
    assert_eq!(files.len(), 2);
    for file in &files {
        assert!(hound::WavReader::open(file).unwrap().duration() > 0);
    }
    std::fs::remove_dir_all(&record_dir).unwrap();
}
