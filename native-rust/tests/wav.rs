use voice_changer::{config::Settings, engine::Engine, wav};
#[test]
fn wav_downmix_preserves_tail_length_and_same_path_is_rejected() {
    let directory = std::env::temp_dir().join(format!("vc-wav-test-{}", std::process::id()));
    std::fs::create_dir_all(&directory).unwrap();
    let input = directory.join("input.wav");
    let output = directory.join("output.wav");
    let mut writer = hound::WavWriter::create(
        &input,
        hound::WavSpec {
            channels: 2,
            sample_rate: 44100,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        },
    )
    .unwrap();
    for _ in 0..10001 {
        writer.write_sample(16384_i16).unwrap();
        writer.write_sample(0_i16).unwrap();
    }
    writer.finalize().unwrap();
    let mut engine = Engine::new(None);
    wav::convert(&mut engine, &input, &output, &Settings::default()).unwrap();
    let mut reader = hound::WavReader::open(&output).unwrap();
    assert_eq!(reader.spec().channels, 1);
    assert_eq!(reader.spec().sample_rate, 44100);
    let samples: Vec<_> = reader.samples::<i16>().map(Result::unwrap).collect();
    assert_eq!(samples.len(), 10001);
    assert!(samples.iter().all(|v| (*v as i32 - 8192).abs() <= 1));
    assert!(wav::convert(&mut engine, &input, &input, &Settings::default()).is_err());
    std::fs::remove_dir_all(directory).unwrap();
}
