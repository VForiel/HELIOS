export default {
    app: {
        title: "HELIOS",
        subtitle: "Visual Architect"
    },
    toolbar: {
        toggleSidebar: "Toggle Sidebar",
        support: "Support the Project",
        github: "GitHub Repository",
        documentation: "Project Documentation",
        import: "Import Pipeline",
        export: "Export Pipeline",
        getCode: "Get Python Code",
        toggleTheme: "Toggle Theme",
        runPipeline: "Run Pipeline",
        language: "Language"
    },
    sidebar: {
        generation: "Generation",
        sampling: "Sampling",
        bulkOptics: "Bulk Optics",
        photonics: "Photonics",
        detection: "Detection",
        result: "Result",
        running: "Running Simulation...",
        downloadImage: "Download Image"
    },
    elements: {
        scene: "Scene Source",
        atmosphere: "Atmosphere",
        telescope: "Telescope",
        lens: "Lens",
        beamSplitter: "Beam Splitter",
        coronagraph: "Coronagraph",
        fiberIn: "Fiber Input",
        fiberOut: "Fiber Output",
        mmi: "MMI Coupler",
        camera: "Camera"
    },
    validation: {
        selfLoop: "A node cannot connect to itself",
        invalidConnection: "Invalid connection",
        generationLayer: "Generation layers (Scene, Atmosphere) can only connect to other generation layers or sampling layers (Telescope)",
        samplingLayer: "Sampling layers (Telescope) can only connect to optical layers (Lens, etc.) or detection layers (Camera)",
        opticalLayer: "Optical layers (Lens, Coronagraph, etc.) can only connect to other optical layers or detection layers (Camera)",
        detectionLayer: "Detection layers (Camera) can only connect to data processing layers",
        dataLayer: "Data layers can only connect to other data layers"
    },
    config: {
        focalLength: "Focal Length",
        splitRatio: "Split Ratio",
        maskType: "Mask Type",
        modes: "Modes",
        inputs: "Inputs",
        outputs: "Outputs",
        fourQuadrants: "4-Quadrants",
        vortex: "Vortex"
    },
    actions: {
        add: "Add",
        remove: "Remove",
        close: "Close",
        download: "Download",
        copy: "Copy",
        paste: "Paste",
        delete: "Delete",
        undo: "Undo",
        redo: "Redo"
    },
    nodes: {
        generation: "Generation",
        sampling: "Sampling",
        bulkOptics: "Bulk Optics",
        photonics: "Photonics",
        detection: "Detection",
        elements: "Elements",
        in: "In",
        out: "Out",
        dropHere: "Drop items here to add elements...",
        deleteLayer: "Delete Layer",
        inputConnection: "Input Connection",
        outputConnection: "Output Connection"
    }
};
