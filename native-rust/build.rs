//! Embeds the app icon in the Windows executable, so Explorer, the taskbar and
//! shortcuts show it.
fn main() {
    println!("cargo:rerun-if-changed=assets/icon.ico");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows") {
        let mut resource = winresource::WindowsResource::new();
        resource.set_icon("assets/icon.ico");
        resource.set("FileDescription", "Voice Changer");
        resource.set("ProductName", "Voice Changer");
        if let Err(error) = resource.compile() {
            panic!("Embed the Windows icon: {error}");
        }
    }
}
