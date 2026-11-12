use voice_changer_native::{
    config::Settings,
    engine::{Converter, Engine},
};
struct InvalidConverter;
impl Converter for InvalidConverter {
    fn process(&mut self, audio: &[f32], _: u32, _: &Settings) -> anyhow::Result<Vec<f32>> {
        Ok(vec![f32::NAN; audio.len()])
    }
    fn reset(&mut self) {}
}
#[test]
fn rejects_nonfinite_inference_and_missing_models() {
    let settings = Settings {
        bypass: false,
        ..Settings::default()
    };
    assert!(Engine::new(None).process(&[0.1], 48000, &settings).is_err());
    assert!(Engine::new(Some(Box::new(InvalidConverter)))
        .process(&[0.1], 48000, &settings)
        .is_err());
}
#[test]
fn applies_gains_and_limits_output() {
    let settings = Settings {
        input_gain: 2.0,
        output_gain: 2.0,
        ..Settings::default()
    };
    assert_eq!(
        Engine::new(None)
            .process(&[0.1, -0.9], 48000, &settings)
            .unwrap(),
        vec![0.4, -1.0]
    );
}
#[test]
fn validates_loaded_presets() {
    let settings = Settings {
        overlap_ms: 0,
        ..Settings::default()
    };
    assert!(settings.validate().is_err());
    let settings = Settings {
        semitones: f32::INFINITY,
        ..Settings::default()
    };
    assert!(settings.validate().is_err());
}
