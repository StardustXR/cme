use std::{os::fd::OwnedFd, sync::Arc};

use stardust_xr_fusion::{
    ClientHandle,
    drawable::{DmatexPlane, DmatexSize, import_dmatex},
};
use timeline_syncobj::timeline_syncobj::TimelineSyncObj;
use tracing::{error, info, warn};
use vulkano::{
    device::{Device, DeviceExtensions, DeviceFeatures},
    image::{
        Image, ImageCreateFlags, ImageCreateInfo, ImageTiling, ImageType, ImageUsage, sys::RawImage,
    },
    instance::InstanceExtensions,
    memory::{
        DedicatedAllocation, DeviceMemory, ExternalMemoryHandleType, ExternalMemoryHandleTypes,
        MemoryAllocateInfo, MemoryPropertyFlags, ResourceMemory,
    },
};

use crate::{format::DmatexFormat, render_device::RenderDevice};

pub struct Dmatex {
    pub image: Arc<Image>,
    pub timeline: TimelineSyncObj,
    pub dmatex_id: u64,
    _client: Arc<ClientHandle>,
}
impl Dmatex {
    // TODO: error handling
    pub fn new(
        client: &Arc<ClientHandle>,
        dev: &Arc<Device>,
        render_dev: &RenderDevice,
        size: DmatexSize,
        format: &DmatexFormat,
        array_layers: Option<u32>,
        usage: ImageUsage,
    ) -> Self {
        let modifiers = dev
            .physical_device()
            .format_properties(format.vk_format())
            .unwrap()
            .drm_format_modifier_properties
            .into_iter()
            .map(|v| v.drm_format_modifier)
            .filter(|modifier| format.variants().iter().any(|v| v.modifier == *modifier))
            .collect::<Vec<_>>();
        let raw_image = RawImage::new(
            dev.clone(),
            ImageCreateInfo {
                flags: ImageCreateFlags::empty(),
                image_type: match &size {
                    DmatexSize::Dim1D(_) => ImageType::Dim1d,
                    DmatexSize::Dim2D(_) => ImageType::Dim2d,
                    DmatexSize::Dim3D(_) => ImageType::Dim3d,
                },
                format: format.vk_format(),
                view_formats: vec![],
                extent: match &size {
                    DmatexSize::Dim1D(v) => [*v, 1, 1],
                    DmatexSize::Dim2D(v) => [v.x, v.y, 1],
                    DmatexSize::Dim3D(v) => (*v).into(),
                },
                array_layers: array_layers.unwrap_or(1),
                tiling: ImageTiling::DrmFormatModifier,
                usage,
                drm_format_modifiers: modifiers,
                external_memory_handle_types: ExternalMemoryHandleTypes::DMA_BUF,
                ..Default::default()
            },
        )
        .unwrap();
        let (modifier, planes) = raw_image.drm_format_modifier().unwrap();
        let mem_reqs = raw_image.memory_requirements();
        info!("modifier {modifier} needs {planes} planes");
        let mems = mem_reqs
            .iter()
            .map(|v| {
                let wants_decicated =
                    v.prefers_dedicated_allocation || v.requires_dedicated_allocation;
                if !wants_decicated {
                    warn!("dmatex image doesn't want a dedicated alloc, too bad");
                }
                let Some((type_index, _)) = dev
                    .physical_device()
                    .memory_properties()
                    .memory_types
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| v.memory_type_bits & (1 << i) != 0)
                    .find(|(_, p)| {
                        // nvidia doesn't put the device local mem first
                        p.property_flags.contains(MemoryPropertyFlags::DEVICE_LOCAL)
                        // not sure if this is even needed, just in case
                        && !p.property_flags.contains(MemoryPropertyFlags::PROTECTED)
                    })
                else {
                    warn!("unable to find memory type for dmatex plane");
                    return None;
                };
                vulkano::memory::DeviceMemory::allocate(
                    dev.clone(),
                    MemoryAllocateInfo {
                        allocation_size: v.layout.size(),
                        memory_type_index: type_index as u32,
                        dedicated_allocation: Some(DedicatedAllocation::Image(&raw_image)),
                        export_handle_types: ExternalMemoryHandleTypes::DMA_BUF,
                        ..MemoryAllocateInfo::default()
                    },
                )
                .inspect_err(|err| error!("failed to allocate mem for dmatex plane: {err}"))
                .ok()
            })
            .collect::<Option<Vec<DeviceMemory>>>();
        let mems = mems.unwrap();
        let fds = mems
            .iter()
            .map(|v| v.export_fd(ExternalMemoryHandleType::DmaBuf))
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        let image = match raw_image.bind_memory(mems.into_iter().map(ResourceMemory::new_dedicated))
        {
            Ok(v) => v,
            Err((err, _, _)) => panic!("failed to bind image mem: {err}"),
        };
        let timeline = TimelineSyncObj::create(render_dev.drm_node()).unwrap();
        let dmatex_id = client.generate_id();
        let first_fd = fds[0].try_clone().unwrap();
        let planes = fds
            .into_iter()
            .chain([first_fd])
            .enumerate()
            .map(|(i, v)| {
                let aspect = match i {
                    0 => vulkano::image::ImageAspect::MemoryPlane0,
                    1 => vulkano::image::ImageAspect::MemoryPlane1,
                    2 => vulkano::image::ImageAspect::MemoryPlane2,
                    3 => vulkano::image::ImageAspect::MemoryPlane3,
                    _ => vulkano::image::ImageAspect::Color,
                };
                let layout = image.subresource_layout(aspect, 0, 0).unwrap();
                DmatexPlane {
                    dmabuf_fd: OwnedFd::from(v).into(),
                    offset: layout.offset as u32,
                    row_size: layout.row_pitch as u32,
                    array_element_size: layout.array_pitch.unwrap_or(0) as u32,
                    depth_slice_size: layout.depth_pitch.unwrap_or(0) as u32,
                }
            })
            .collect::<Vec<_>>();
        import_dmatex(
            client,
            dmatex_id,
            size,
            format.drm_fourcc() as u32,
            modifier,
            format!("{:?}", format.vk_format()).contains("SRGB"),
            array_layers,
            &planes,
            timeline.export().unwrap().into(),
        )
        .unwrap();

        Self {
            image: Arc::new(image),
            timeline,
            dmatex_id,
            _client: client.clone(),
        }
    }
}

impl Dmatex {
    /// empty, exists just incase any instance exts are required in the future
    pub const fn required_instance_exts() -> InstanceExtensions {
        InstanceExtensions::empty()
    }
    pub const fn required_device_exts() -> DeviceExtensions {
        DeviceExtensions {
            ext_image_drm_format_modifier: true,
            ext_external_memory_dma_buf: true,
            khr_external_memory: true,
            khr_external_memory_fd: true,
            khr_external_semaphore: true,
            khr_external_semaphore_fd: true,

            ..DeviceExtensions::empty()
        }
    }
    /// empty, exists just incase any device features are required in the future
    pub const fn required_device_features() -> DeviceFeatures {
        DeviceFeatures::empty()
    }
}
