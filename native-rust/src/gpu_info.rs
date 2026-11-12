//! Names the GPU each device runs on, and picks the GPU DirectML should use on
//! laptops with two (it would otherwise take the first, often the integrated one).
//! Detection runs once in the background through wgpu, which the UI already uses.
use crate::config::Device;
use std::sync::OnceLock;

#[derive(Clone, Debug)]
pub struct Gpu {
    pub name: String,
    vendor: u32,
    discrete: bool,
    /// Position among the DirectX 12 adapters, which is DirectML's device id.
    dx12_index: Option<usize>,
}

const NVIDIA: u32 = 0x10de;
const AMD: u32 = 0x1002;

static GPUS: OnceLock<Vec<Gpu>> = OnceLock::new();

/// Starts detection; `gpus` returns nothing until it finishes.
pub fn detect_in_background() {
    let _ = std::thread::Builder::new()
        .name("gpu-detect".into())
        .spawn(|| {
            let found = std::panic::catch_unwind(enumerate).unwrap_or_default();
            if found.is_empty() {
                log::info!("No GPU found");
            }
            for gpu in &found {
                log::info!(
                    "GPU: {} ({}{})",
                    gpu.name,
                    if gpu.discrete {
                        "discrete"
                    } else {
                        "integrated"
                    },
                    gpu.dx12_index
                        .map(|index| format!(", DirectX 12 adapter {index}"))
                        .unwrap_or_default()
                );
            }
            if let Some(index) = directml_choice(&found) {
                log::info!("DirectML will use DirectX 12 adapter {index}");
                crate::runtime::set_directml_device(index as i32);
            }
            let _ = GPUS.set(found);
        });
}

pub fn gpus() -> &'static [Gpu] {
    GPUS.get().map_or(&[], Vec::as_slice)
}

fn enumerate() -> Vec<Gpu> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let mut dx12 = 0;
    let mut found: Vec<Gpu> = Vec::new();
    for adapter in instance.enumerate_adapters(wgpu::Backends::PRIMARY) {
        let info = adapter.get_info();
        let dx12_index = (info.backend == wgpu::Backend::Dx12).then(|| {
            dx12 += 1;
            dx12 - 1
        });
        if matches!(
            info.device_type,
            wgpu::DeviceType::Cpu | wgpu::DeviceType::Other
        ) || info.name.is_empty()
        {
            continue;
        }
        // One GPU shows up once per backend (Vulkan and DX12 on Windows); keep it once.
        match found.iter_mut().find(|gpu| gpu.name == info.name) {
            Some(gpu) => gpu.dx12_index = gpu.dx12_index.or(dx12_index),
            None => found.push(Gpu {
                name: info.name.trim().to_owned(),
                vendor: info.vendor,
                discrete: info.device_type == wgpu::DeviceType::DiscreteGpu,
                dx12_index,
            }),
        }
    }
    found
}

/// The DirectX 12 adapter DirectML should use: a discrete NVIDIA or AMD GPU first.
fn directml_choice(gpus: &[Gpu]) -> Option<usize> {
    let rank = |gpu: &Gpu| (gpu.discrete, gpu.vendor == NVIDIA || gpu.vendor == AMD);
    gpus.iter()
        .filter(|gpu| gpu.dx12_index.is_some())
        .max_by_key(|gpu| rank(gpu))
        .and_then(|gpu| gpu.dx12_index)
}

/// The GPU `device` runs on, once detection has finished.
pub fn name_for(device: Device) -> Option<&'static str> {
    let gpus = gpus();
    let gpu = match device {
        Device::Cpu => None,
        Device::Gpu => gpus.iter().find(|gpu| gpu.vendor == NVIDIA),
        Device::DirectMl => {
            let index = directml_choice(gpus)?;
            gpus.iter().find(|gpu| gpu.dx12_index == Some(index))
        }
    };
    gpu.map(|gpu| gpu.name.as_str())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gpu(name: &str, vendor: u32, discrete: bool, dx12_index: Option<usize>) -> Gpu {
        Gpu {
            name: name.into(),
            vendor,
            discrete,
            dx12_index,
        }
    }

    #[test]
    fn directml_prefers_the_discrete_gpu_on_hybrid_laptops() {
        let laptop = [
            gpu("Intel(R) Graphics", 0x8086, false, Some(0)),
            gpu(
                "NVIDIA GeForce RTX 5070 Ti Laptop GPU",
                NVIDIA,
                true,
                Some(1),
            ),
        ];
        assert_eq!(directml_choice(&laptop), Some(1));
        let single = [gpu("AMD Radeon 780M", AMD, false, Some(0))];
        assert_eq!(directml_choice(&single), Some(0));
        assert_eq!(directml_choice(&[gpu("x", NVIDIA, true, None)]), None);
    }
}
