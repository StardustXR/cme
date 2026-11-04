use std::sync::Arc;

use stardust_xr_fusion::{
    client::{Client, ClientHandler},
    dmatex::{self, AlphaMode, DmatexPlane, DmatexPlanes, DmatexRef, DmatexSize},
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
        DedicatedAllocation, ExternalMemoryHandleType, ExternalMemoryHandleTypes,
        MemoryAllocateInfo, MemoryPropertyFlags, ResourceMemory,
    },
};

use crate::{format::DmatexFormat, render_device::RenderDevice};

#[derive(Debug)]
pub struct Dmatex {
    pub image: Arc<Image>,
    pub timeline: TimelineSyncObj,
    pub dmatex: DmatexRef,
    /// Set for [`Dmatex::new_host_linear`] images: the persistently mapped plane
    pub host_map: Option<HostMap>,
}

/// A persistently mapped, host-coherent LINEAR plane the CPU can write directly
#[derive(Debug, Clone, Copy)]
pub struct HostMap {
    ptr: usize,
    len: usize,
    pub offset: usize,
    pub row_pitch: usize,
}
impl HostMap {
    /// Copy `data` into pixel row `row`. Returns `false` (and writes nothing) if the
    /// row would land outside the mapping.
    ///
    /// # Safety
    /// The [`Dmatex`] this came from must still be alive. The remote process may be
    /// sampling the image at the same time, so callers must tolerate tearing, and
    /// must not write the same mapping from several threads at once.
    pub unsafe fn write_row(&self, row: usize, data: &[u8]) -> bool {
        let Some(start) = row
            .checked_mul(self.row_pitch)
            .and_then(|r| r.checked_add(self.offset))
        else {
            return false;
        };
        if start
            .checked_add(data.len())
            .is_none_or(|end| end > self.len)
        {
            return false;
        }
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr(),
                (self.ptr as *mut u8).add(start),
                data.len(),
            );
        }
        true
    }
}
impl Dmatex {
    // TODO: error handling
    pub async fn new(
        client: &Client<impl ClientHandler>,
        dev: &Arc<Device>,
        render_dev: &RenderDevice,
        size: DmatexSize,
        format: &DmatexFormat,
        array_layers: Option<u32>,
        alpha_mode: AlphaMode,
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
                    DmatexSize::Size1D { size: _ } => ImageType::Dim1d,
                    DmatexSize::Size2D { size: _ } => ImageType::Dim2d,
                    DmatexSize::Size3D { size: _ } => ImageType::Dim3d,
                },
                format: format.vk_format(),
                view_formats: vec![],
                extent: match &size {
                    DmatexSize::Size1D { size: v } => [*v, 1, 1],
                    DmatexSize::Size2D { size: v } => [v.x, v.y, 1],
                    DmatexSize::Size3D { size: v } => (*v).into(),
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
        // this doesn't support disjoint
        let mem_reqs = raw_image.memory_requirements();
        if mem_reqs.len() != 1 {
            panic!(
                "dmatex had more than one memory requirement: {}",
                mem_reqs.len()
            );
        }
        let mem_req = mem_reqs[0];
        info!("modifier {modifier} needs {planes} planes");
        let wants_decicated =
            mem_req.prefers_dedicated_allocation || mem_req.requires_dedicated_allocation;
        if !wants_decicated {
            warn!("dmatex image doesn't want a dedicated alloc, too bad");
        }
        let Some((type_index, _)) = dev
            .physical_device()
            .memory_properties()
            .memory_types
            .iter()
            .enumerate()
            .filter(|(i, _)| mem_req.memory_type_bits & (1 << i) != 0)
            .find(|(_, p)| {
                // nvidia doesn't put the device local mem first
                p.property_flags.contains(MemoryPropertyFlags::DEVICE_LOCAL)
                        // not sure if this is even needed, just in case
                        && !p.property_flags.contains(MemoryPropertyFlags::PROTECTED)
            })
        else {
            panic!("unable to find memory type for dmatex plane");
        };
        let mem = vulkano::memory::DeviceMemory::allocate(
            dev.clone(),
            MemoryAllocateInfo {
                allocation_size: mem_req.layout.size(),
                memory_type_index: type_index as u32,
                dedicated_allocation: Some(DedicatedAllocation::Image(&raw_image)),
                export_handle_types: ExternalMemoryHandleTypes::DMA_BUF,
                ..MemoryAllocateInfo::default()
            },
        )
        .inspect_err(|err| error!("failed to allocate mem for dmatex plane: {err}"))
        .unwrap();
        let dmabuf_fd = mem
            .export_fd(ExternalMemoryHandleType::DmaBuf)
            .unwrap()
            .into();
        let image = match raw_image.bind_memory([ResourceMemory::new_dedicated(mem)]) {
            Ok(v) => v,
            Err((err, _, _)) => panic!("failed to bind image mem: {err}"),
        };
        let timeline = TimelineSyncObj::new(render_dev.drm_node()).unwrap();
        let planes = (0..planes)
            .map(|i| {
                let aspect = match i {
                    0 => vulkano::image::ImageAspect::MemoryPlane0,
                    1 => vulkano::image::ImageAspect::MemoryPlane1,
                    2 => vulkano::image::ImageAspect::MemoryPlane2,
                    3 => vulkano::image::ImageAspect::MemoryPlane3,
                    _ => vulkano::image::ImageAspect::Color,
                };
                let layout = image.subresource_layout(aspect, 0, 0).unwrap();
                DmatexPlane {
                    offset: layout.offset,
                    row_size: layout.row_pitch,
                    array_element_size: layout.array_pitch.unwrap_or(0),
                    depth_slice_size: layout.depth_pitch.unwrap_or(0),
                }
            })
            .collect::<Vec<_>>();
        let dmatex = client
            .dmatex_interface()
            .import_dmatex(
                size,
                dmatex::DmatexFormat {
                    drm_fourcc: format.drm_fourcc() as u32,
                    drm_modifier: modifier,
                    is_srgb: format!("{:?}", format.vk_format()).contains("SRGB"),
                    alpha_mode,
                    ycbcr_info: None,
                },
                array_layers.unwrap_or(1),
                DmatexPlanes::Simple { dmabuf_fd, planes },
                timeline.export().unwrap(),
            )
            .await
            .unwrap()
            .unwrap();

        Self {
            image: Arc::new(image),
            timeline,
            dmatex,
            host_map: None,
        }
    }

    /// A 2D LINEAR dmatex in host-visible, host-coherent memory that the CPU writes
    /// through [`Dmatex::host_map`], then signals `timeline` itself. Uploading this way
    /// needs no queue submit at all, which is useful for shm clients and works around
    /// drivers that lose the device on transfer submits (NVIDIA 615.78, Xid 32).
    ///
    /// Prefers ReBAR (DEVICE_LOCAL | HOST_VISIBLE) memory and falls back to system memory
    /// when that allocation fails.
    pub async fn new_host_linear(
        client: &Client<impl ClientHandler>,
        dev: &Arc<Device>,
        render_dev: &RenderDevice,
        size: [u32; 2],
        format: &DmatexFormat,
        alpha_mode: AlphaMode,
    ) -> Result<Self, String> {
        if !format.variants().iter().any(|v| v.modifier == 0) {
            return Err("server doesn't accept LINEAR".into());
        }
        let raw_image = RawImage::new(
            dev.clone(),
            ImageCreateInfo {
                flags: ImageCreateFlags::empty(),
                image_type: ImageType::Dim2d,
                format: format.vk_format(),
                view_formats: vec![],
                extent: [size[0], size[1], 1],
                array_layers: 1,
                tiling: ImageTiling::DrmFormatModifier,
                usage: ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED,
                // two entries so vulkano uses the list create info (one entry means explicit layout)
                drm_format_modifiers: vec![0, 0],
                external_memory_handle_types: ExternalMemoryHandleTypes::DMA_BUF,
                ..Default::default()
            },
        )
        .map_err(|e| format!("linear image create: {e}"))?;
        let (modifier, planes) = raw_image.drm_format_modifier().ok_or("no modifier")?;
        if modifier != 0 || planes != 1 {
            return Err(format!("got modifier {modifier} planes {planes}"));
        }
        let mem_reqs = raw_image.memory_requirements();
        let mem_req = mem_reqs[0];
        let types = &dev.physical_device().memory_properties().memory_types;
        let pick = |want: MemoryPropertyFlags| {
            types
                .iter()
                .enumerate()
                .filter(|(i, _)| mem_req.memory_type_bits & (1 << i) != 0)
                .find(|(_, p)| {
                    p.property_flags.contains(want)
                        && !p.property_flags.contains(MemoryPropertyFlags::PROTECTED)
                })
                .map(|(i, _)| i)
        };
        let hv = MemoryPropertyFlags::HOST_VISIBLE | MemoryPropertyFlags::HOST_COHERENT;
        // ReBAR (DEVICE_LOCAL|HV) first: the server's import of sysmem dmabufs fails
        // intermittently (InternalImportError). Fall back to sysmem when BAR is full.
        let bar = pick(hv | MemoryPropertyFlags::DEVICE_LOCAL);
        let sys = pick(hv);
        let order: Vec<usize> = bar.into_iter().chain(sys).collect();
        if order.is_empty() {
            return Err(format!(
                "no host visible mem type in bits {:#x}",
                mem_req.memory_type_bits
            ));
        }
        let mut last_err = String::new();
        let mut found = None;
        for type_index in order {
            match vulkano::memory::DeviceMemory::allocate(
                dev.clone(),
                MemoryAllocateInfo {
                    allocation_size: mem_req.layout.size(),
                    memory_type_index: type_index as u32,
                    dedicated_allocation: Some(DedicatedAllocation::Image(&raw_image)),
                    export_handle_types: ExternalMemoryHandleTypes::DMA_BUF,
                    ..MemoryAllocateInfo::default()
                },
            ) {
                Ok(m) => {
                    tracing::debug!(
                        "host-linear dmatex {}x{} mem type {type_index} size {}",
                        size[0],
                        size[1],
                        mem_req.layout.size()
                    );
                    found = Some(m);
                    break;
                }
                Err(e) => {
                    tracing::warn!("host-linear alloc in mem type {type_index} failed: {e}");
                    last_err = format!("alloc: {e}");
                }
            }
        }
        let mut mem = found.ok_or(last_err)?;
        mem.map(vulkano::memory::MemoryMapInfo {
            offset: 0,
            size: mem.allocation_size(),
            ..Default::default()
        })
        .map_err(|e| format!("map: {e}"))?;
        let ms = mem.mapping_state().ok_or("not mapped")?;
        let (ptr, len) = (ms.ptr().as_ptr() as usize, ms.size() as usize);
        let dmabuf_fd = mem
            .export_fd(ExternalMemoryHandleType::DmaBuf)
            .map_err(|e| format!("export: {e}"))?
            .into();
        let image = raw_image
            .bind_memory([ResourceMemory::new_dedicated(mem)])
            .map_err(|(e, _, _)| format!("bind: {e}"))?;
        let layout = image
            .subresource_layout(vulkano::image::ImageAspect::MemoryPlane0, 0, 0)
            .map_err(|e| format!("layout: {e}"))?;
        let timeline =
            TimelineSyncObj::new(render_dev.drm_node()).map_err(|e| format!("timeline: {e}"))?;
        let dmatex = client
            .dmatex_interface()
            .import_dmatex(
                DmatexSize::Size2D { size: size.into() },
                dmatex::DmatexFormat {
                    drm_fourcc: format.drm_fourcc() as u32,
                    drm_modifier: 0,
                    is_srgb: format!("{:?}", format.vk_format()).contains("SRGB"),
                    alpha_mode,
                    ycbcr_info: None,
                },
                1u32,
                DmatexPlanes::Simple {
                    dmabuf_fd,
                    planes: vec![DmatexPlane {
                        offset: layout.offset,
                        row_size: layout.row_pitch,
                        array_element_size: 0,
                        depth_slice_size: 0,
                    }],
                },
                timeline
                    .export()
                    .map_err(|e| format!("timeline export: {e}"))?,
            )
            .await
            .map_err(|e| format!("import ipc: {e}"))?
            .map_err(|e| format!("import: {e:?}"))?;
        Ok(Self {
            image: Arc::new(image),
            timeline,
            dmatex,
            host_map: Some(HostMap {
                ptr,
                len,
                offset: layout.offset as usize,
                row_pitch: layout.row_pitch as usize,
            }),
        })
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
