/**
 * Icon mapping utility for HELIOS elements
 * Maps element types to their corresponding SVG icon paths
 */

// Map of element types to icon file names
const ICON_MAP = {
    // Scene and celestial objects
    scene: 'scene.svg',

    // Optical elements
    telescope: 'telescope.svg',
    atmosphere: 'atmosphere.svg',
    adaptive_optics: 'adaptive_optics.svg',
    coronagraph: 'coronagraph.svg',
    beam_splitter: 'beam_splitter.svg',

    // Detectors
    camera: 'camera.svg',

    // Fiber optics
    fiber_in: 'fiber_in.svg',
    fiber_out: 'fiber_out.svg',

    // Photonics
    photonic: 'photonic_chip.svg',
    photonic_chip: 'photonic_chip.svg',
    phase_shifter: 'phase_shifter.svg',
    tops: 'phase_shifter.svg', // TOPS uses phase shifter icon
    mmi: 'mmi.svg',
    swap: 'swap.svg',
    y_splitter: 'splitter.svg',

    // Interferometry
    interferometer: 'interferometer.svg',

    // Generic fallback
    lens: 'lens.svg',
};

/**
 * Get the icon path for a given element type
 * @param {string} type - The element type
 * @returns {string} - The path to the icon file
 */
export function getElementIcon(type) {
    const iconFile = ICON_MAP[type] || 'beam_splitter.svg'; // Default fallback
    return `/icons/${iconFile}`;
}

/**
 * Get icon path for photonic sub-components
 * @param {string} photonicType - The photonic component type
 * @returns {string} - The path to the icon file
 */
export function getPhotonicIcon(photonicType) {
    const typeMap = {
        'y_splitter': 'splitter.svg',
        'tops': 'phase_shifter.svg',
        'mmi': 'mmi.svg',
        'swap': 'swap.svg',
    };

    const iconFile = typeMap[photonicType] || 'photonic_chip.svg';
    return `/icons/${iconFile}`;
}

export default getElementIcon;
