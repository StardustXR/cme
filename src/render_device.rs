use std::sync::Arc;

use stardust_xr_fusion::client::{Client, ClientHandler};
use thiserror::Error;
use timeline_syncobj::render_node::DrmRenderNode;
use vulkano::{VulkanError, device::physical::PhysicalDevice, instance::Instance};

use crate::get_phys_dev_node_id;

/// Roughly corresponds to a GPU
pub struct RenderDevice {
    drm_node: DrmRenderNode,
    render_node_id: u64,
}

impl RenderDevice {
    /// initializes Self with the preferred [`RenderDevice`] of the server
    pub async fn primary_server_device(
        client: &Arc<Client<impl ClientHandler>>,
    ) -> Result<Self, RenderDeviceCreationError> {
        let id = client
            .dmatex_interface()
            .primary_render_node_id()
            .await
            .map_err(|_| RenderDeviceCreationError::FailedToGetDeviceId)?;
        let drm_node =
            DrmRenderNode::new(id).map_err(RenderDeviceCreationError::FailedToOpenDrmNode)?;

        Ok(Self {
            drm_node,
            render_node_id: id,
        })
    }

    pub fn get_physical_device(
        &self,
        instance: &Arc<Instance>,
    ) -> Result<Arc<PhysicalDevice>, RenderDevicePhysDevError> {
        instance
            .enumerate_physical_devices()
            .map_err(RenderDevicePhysDevError::FailedToEnumeratePhysDevs)?
            .find(|p| get_phys_dev_node_id(p) == self.render_node_id)
            .ok_or(RenderDevicePhysDevError::FailedToFindPhysDev)
    }
    pub fn drm_node_id(&self) -> u64 {
        self.render_node_id
    }
    pub fn drm_node(&self) -> &DrmRenderNode {
        &self.drm_node
    }
}

#[derive(Debug, Error)]
pub enum RenderDeviceCreationError {
    #[error("failed to get the RenderDevice id from the server")]
    FailedToGetDeviceId,
    #[error("unable to open DrmRenderNode: {0}")]
    FailedToOpenDrmNode(rustix::io::Errno),
}
#[derive(Debug, Error)]
pub enum RenderDevicePhysDevError {
    #[error("failed to enumerate Physical Devices: {0}")]
    FailedToEnumeratePhysDevs(VulkanError),
    #[error("failed to find matching physical device")]
    FailedToFindPhysDev,
}
