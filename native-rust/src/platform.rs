use anyhow::Result;

/// Permission prompts run on the UI thread. The user can retry Start after granting access.
pub fn microphone_ready() -> Result<bool> {
    #[cfg(target_os = "android")]
    {
        android_permission()
    }
    #[cfg(target_os = "ios")]
    {
        ios_permission()
    }
    #[cfg(not(any(target_os = "android", target_os = "ios")))]
    {
        Ok(true)
    }
}

#[cfg(target_os = "android")]
static ANDROID_APP: std::sync::OnceLock<android_activity::AndroidApp> = std::sync::OnceLock::new();

#[cfg(target_os = "android")]
pub fn android_init(app: android_activity::AndroidApp) {
    // Resolve relative model paths inside app-accessible storage, never the filesystem root.
    if let Some(path) = app
        .external_data_path()
        .or_else(|| app.internal_data_path())
    {
        let _ = std::env::set_current_dir(path);
    }
    let _ = ANDROID_APP.set(app);
}

#[cfg(target_os = "android")]
fn android_permission() -> Result<bool> {
    use jni::{
        objects::{JObject, JValue},
        JavaVM,
    };
    let app = ANDROID_APP
        .get()
        .ok_or_else(|| anyhow::anyhow!("Android activity is not initialized"))?;
    // AndroidApp owns the VM and activity for the lifetime of android_main.
    let vm = unsafe { JavaVM::from_raw(app.vm_as_ptr().cast()) }?;
    let mut env = vm.attach_current_thread()?;
    let activity = unsafe { JObject::from_raw(app.activity_as_ptr().cast()) };
    let permission = JObject::from(env.new_string("android.permission.RECORD_AUDIO")?);
    let granted = env
        .call_method(
            &activity,
            "checkSelfPermission",
            "(Ljava/lang/String;)I",
            &[JValue::Object(&permission)],
        )?
        .i()?
        == 0;
    if !granted {
        let activity_app = app.clone();
        app.run_on_java_main_thread(Box::new(move || {
            let request = || -> Result<()> {
                let vm = unsafe { JavaVM::from_raw(activity_app.vm_as_ptr().cast()) }?;
                let mut env = vm.attach_current_thread()?;
                let activity = unsafe { JObject::from_raw(activity_app.activity_as_ptr().cast()) };
                let permission = JObject::from(env.new_string("android.permission.RECORD_AUDIO")?);
                let permissions = env.new_object_array(1, "java/lang/String", &permission)?;
                env.call_method(
                    &activity,
                    "requestPermissions",
                    "([Ljava/lang/String;I)V",
                    &[JValue::Object(&permissions), JValue::Int(1)],
                )?;
                Ok(())
            };
            if let Err(error) = request() {
                eprintln!("Microphone permission: {error:#}");
            }
        }));
    }
    Ok(granted)
}

#[cfg(target_os = "ios")]
#[allow(deprecated)]
fn ios_permission() -> Result<bool> {
    use objc2_avf_audio::{AVAudioSession, AVAudioSessionRecordPermission};
    let session = unsafe { AVAudioSession::sharedInstance() };
    let status = unsafe { session.recordPermission() };
    if status == AVAudioSessionRecordPermission::Granted {
        return Ok(true);
    }
    if status == AVAudioSessionRecordPermission::Denied {
        anyhow::bail!("Microphone permission denied. Enable it in iOS Settings.");
    }
    let completion = block2::RcBlock::new(|_granted: objc2::runtime::Bool| {});
    unsafe {
        session.requestRecordPermission(&completion);
    }
    Ok(false)
}

pub fn prepare_audio_session() -> Result<()> {
    #[cfg(target_os = "ios")]
    unsafe {
        use objc2_avf_audio::{
            AVAudioSession, AVAudioSessionCategoryOptions, AVAudioSessionCategoryPlayAndRecord,
            AVAudioSessionModeMeasurement,
        };
        let session = AVAudioSession::sharedInstance();
        session
            .setCategory_mode_options_error(
                AVAudioSessionCategoryPlayAndRecord,
                AVAudioSessionModeMeasurement,
                AVAudioSessionCategoryOptions::DefaultToSpeaker,
            )
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        session
            .setActive_error(true)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
    }
    Ok(())
}
