use std::{collections::HashMap, sync::Arc};

use drm_fourcc::DrmFourcc;
use stardust_xr_fusion::{ClientHandle, drawable::enumerate_dmatex_formats, node::NodeResult};
use tracing::{error, warn};
use vulkano::format::Format;

use crate::render_device::RenderDevice;

// TODO: Docs
#[derive(Debug, Clone)]
pub struct DmatexFormat {
    format: Format,
    fourcc: DrmFourcc,
    variants: Vec<DmatexFormatVariant>,
}
impl DmatexFormat {
    pub fn vk_format(&self) -> Format {
        self.format
    }
    pub fn drm_fourcc(&self) -> DrmFourcc {
        self.fourcc
    }
    pub fn variants(&self) -> &[DmatexFormatVariant] {
        &self.variants
    }
}
impl DmatexFormat {
    pub async fn enumerate(
        client: &Arc<ClientHandle>,
        render_device: &RenderDevice,
    ) -> NodeResult<HashMap<Format, DmatexFormat>> {
        let formats = enumerate_dmatex_formats(client, render_device.drm_node_id()).await?;
        let mut out = HashMap::new();
        for v in formats {
            let Ok(fourcc) = drm_fourcc::DrmFourcc::try_from(v.format) else {
                error!("unable to parse drm_fourcc: {:X}", v.format);
                continue;
            };
            let Some(format) = Format::from_drm_fourcc(fourcc) else {
                warn!("failed to get vulkan format for drm_fourcc: {fourcc}");
                continue;
            };
            let format = if v.is_srgb {
                let Some(format) = format.to_srgb() else {
                    warn!("failed to do srgb conversion for: {format:?}");
                    continue;
                };
                format
            } else {
                format
            };
            out.entry(format)
                .or_insert_with(|| DmatexFormat {
                    format,
                    fourcc,
                    variants: vec![],
                })
                .variants
                .push(DmatexFormatVariant {
                    modifier: v.drm_modifier,
                    planes: v.planes,
                });
        }

        Ok(out)
    }
}
#[derive(Debug, Clone, Copy)]
pub struct DmatexFormatVariant {
    pub modifier: u64,
    pub planes: u32,
}

pub trait VulkanoFormatExtension: Sized {
    fn from_drm_fourcc(drm_format: drm_fourcc::DrmFourcc) -> Option<Self>;
    fn to_srgb(&self) -> Option<Self>;
}
impl VulkanoFormatExtension for Format {
    fn from_drm_fourcc(drm_format: drm_fourcc::DrmFourcc) -> Option<Self> {
        use Format as F;
        use drm_fourcc::DrmFourcc as D;
        Some(match drm_format {
            D::Abgr1555 | D::Xbgr1555 => F::R5G5B5A1_UNORM_PACK16,
            D::Abgr2101010 | D::Xbgr2101010 => F::A2B10G10R10_UNORM_PACK32,
            D::Abgr4444 | D::Xbgr4444 => F::A4B4G4R4_UNORM_PACK16,
            D::Abgr8888 | D::Xbgr8888 => F::R8G8B8A8_UNORM,
            D::Argb1555 | D::Xrgb1555 => F::A1R5G5B5_UNORM_PACK16,
            D::Argb2101010 | D::Xrgb2101010 => F::A2R10G10B10_UNORM_PACK32,
            D::Argb4444 | D::Xrgb4444 => F::B4G4R4A4_UNORM_PACK16,
            D::Argb8888 | D::Xrgb8888 => F::B8G8R8A8_UNORM,
            D::Bgr565 => F::B5G6R5_UNORM_PACK16,
            D::Bgr888 => F::B8G8R8_UNORM,
            // D::Bgr888_a8 => F::B8G8R8A8_UNORM,
            D::Bgra4444 | D::Bgrx4444 => F::B4G4R4A4_UNORM_PACK16,
            D::Bgra5551 | D::Bgrx5551 => F::B5G5R5A1_UNORM_PACK16,
            D::Bgra8888 | D::Bgrx8888 => F::B8G8R8A8_UNORM,
            D::R16 => F::R16_UNORM,
            D::R8 => F::R8_UNORM,
            D::Rg1616 => F::R16G16_UNORM,
            D::Rg88 => F::R8G8_UNORM,
            D::Rgb565 => F::R5G6B5_UNORM_PACK16,
            D::Rgb888 => F::R8G8B8_UNORM,
            // D::Rgb888_a8 => F::R8G8B8A8_UNORM,
            D::Rgba4444 | D::Rgbx4444 => F::R4G4B4A4_UNORM_PACK16,
            D::Rgba5551 | D::Rgbx5551 => F::R5G5B5A1_UNORM_PACK16,
            D::Rgba8888 | D::Rgbx8888 => F::R8G8B8A8_UNORM,
            D::Abgr16161616f => F::R16G16B16A16_SFLOAT,
            _ => return None,
        })
    }
    fn to_srgb(&self) -> Option<Self> {
        use Format as F;
        Some(match self {
            F::R8_UNORM => F::R8_SRGB,
            F::R8G8_UNORM => F::R8G8_SRGB,
            F::R8G8B8_UNORM => F::R8G8B8_SRGB,
            F::B8G8R8_UNORM => F::B8G8R8_SRGB,
            F::R8G8B8A8_UNORM => F::R8G8B8A8_SRGB,
            F::B8G8R8A8_UNORM => F::B8G8R8A8_SRGB,
            F::A8B8G8R8_UNORM_PACK32 => F::A8B8G8R8_SRGB_PACK32,
            F::BC1_RGB_UNORM_BLOCK => F::BC1_RGB_SRGB_BLOCK,
            F::BC1_RGBA_UNORM_BLOCK => F::BC1_RGBA_SRGB_BLOCK,
            F::BC2_UNORM_BLOCK => F::BC2_SRGB_BLOCK,
            F::BC3_UNORM_BLOCK => F::BC3_SRGB_BLOCK,
            F::BC7_UNORM_BLOCK => F::BC7_SRGB_BLOCK,
            F::ETC2_R8G8B8_UNORM_BLOCK => F::ETC2_R8G8B8_SRGB_BLOCK,
            F::ETC2_R8G8B8A1_UNORM_BLOCK => F::ETC2_R8G8B8A1_SRGB_BLOCK,
            F::ETC2_R8G8B8A8_UNORM_BLOCK => F::ETC2_R8G8B8A8_SRGB_BLOCK,
            _ => return None,
        })
    }
}
