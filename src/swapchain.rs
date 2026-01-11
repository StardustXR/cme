use std::{os::fd::AsFd, sync::Arc};

use stardust_xr_fusion::{
    ClientHandle,
    drawable::{DmatexMaterialParam, DmatexSize},
};
use vulkano::{
    device::{Device, Queue, QueueGuard},
    image::{Image, ImageUsage},
    sync::semaphore::{
        ExternalSemaphoreHandleType, ExternalSemaphoreHandleTypes, ImportSemaphoreFdInfo,
        Semaphore, SemaphoreCreateInfo, SemaphoreImportFlags,
    },
};

use crate::{dmatex::Dmatex, format::DmatexFormat, render_device::RenderDevice};

pub struct Swapchain<const IMAGES: usize = 3> {
    images: [(Arc<Dmatex>, u64); IMAGES],
    next_image: usize,
}

impl Swapchain {
    pub fn new(
        client: &Arc<ClientHandle>,
        dev: &Arc<Device>,
        render_dev: &RenderDevice,
        size: DmatexSize,
        format: &DmatexFormat,
        array_layers: Option<u32>,
        usage: ImageUsage,
    ) -> Self {
        let images = [(); _]
            .map(|_| {
                Arc::new(Dmatex::new(
                    client,
                    dev,
                    render_dev,
                    size.clone(),
                    format,
                    array_layers,
                    usage,
                ))
            })
            .map(|v| (v, 0));
        for image in &images {
            unsafe {
                image.0.timeline.signal(0).unwrap();
            }
        }
        Self {
            images,
            next_image: 0,
        }
    }
    pub fn prepare_next_image(&mut self) -> SwapchainFrameHandle {
        let images_len = self.images.len();
        let (image, previous_release) = &mut self.images[self.next_image];
        self.next_image += 1;
        self.next_image %= images_len;
        let acquire_point = *previous_release + 1;
        let previous_server_release = *previous_release;
        *previous_release = acquire_point + 1;
        SwapchainFrameHandle {
            previous_server_release,
            server_acquire: acquire_point,
            next_server_release: *previous_release,
            image: image.clone(),
        }
    }
}
pub struct SwapchainFrameHandle {
    previous_server_release: u64,
    server_acquire: u64,
    next_server_release: u64,
    image: Arc<Dmatex>,
}
impl SwapchainFrameHandle {
    pub fn image(&self) -> Arc<Image> {
        self.image.image.clone()
    }
    pub fn blocking_release_wait(&self) {
        self.image
            .timeline
            .blocking_wait(self.previous_server_release, None)
            .unwrap();
    }
    pub fn submit(
        self,
        dev: &Arc<Device>,
        render_queue: &Arc<Queue>,
        submit: impl FnOnce(Arc<Semaphore>, QueueGuard, Arc<Semaphore>),
    ) -> DmatexMaterialParam {
        let wait_semaphore = Arc::new(Semaphore::from_pool(dev.clone()).unwrap());
        unsafe {
            wait_semaphore
                .import_fd(ImportSemaphoreFdInfo {
                    file: Some(
                        self.image
                            .timeline
                            .export_sync_file_point(self.previous_server_release)
                            .unwrap()
                            .into(),
                    ),
                    flags: SemaphoreImportFlags::TEMPORARY,
                    ..ImportSemaphoreFdInfo::handle_type(ExternalSemaphoreHandleType::SyncFd)
                })
                .unwrap()
        }
        // TODO: custom pool?
        let submit_semaphore = Arc::new(
            Semaphore::new(
                dev.clone(),
                SemaphoreCreateInfo {
                    export_handle_types: ExternalSemaphoreHandleTypes::SYNC_FD,
                    ..Default::default()
                },
            )
            .unwrap(),
        );
        render_queue.with(|guard| submit(wait_semaphore, guard, submit_semaphore.clone()));

        let fd = unsafe {
            submit_semaphore
                .export_fd(ExternalSemaphoreHandleType::SyncFd)
                .unwrap()
        };
        self.image
            .timeline
            .import_sync_file_point(fd.as_fd(), self.server_acquire)
            .unwrap();

        DmatexMaterialParam {
            dmatex_id: self.image.dmatex_id,
            acquire_point: self.server_acquire,
            release_point: self.next_server_release,
        }
    }
}
