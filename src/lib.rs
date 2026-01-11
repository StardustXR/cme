use std::sync::Arc;

use vulkano::device::physical::PhysicalDevice;

pub mod dmatex;
pub mod swapchain;
pub mod format;
pub mod render_device;

pub fn get_phys_dev_node_id(phys_dev: &Arc<PhysicalDevice>) -> u64 {
    let props = phys_dev.properties();
    // Create dev_t from the primary node major/minor numbers
    let major = props.render_major.unwrap() as u64;
    let minor = props.render_minor.unwrap() as u64;
    // On Linux, dev_t is created with makedev(major, minor)
    // which is ((major & 0xfffff000) << 32) | ((major & 0xfff) << 8) | (minor & 0xff)
    ((major & 0xfffff000) << 32) | ((major & 0xfff) << 8) | (minor & 0xff)
}
