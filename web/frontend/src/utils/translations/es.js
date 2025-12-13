export default {
    app: {
        title: "HELIOS",
        subtitle: "Arquitecto Visual"
    },
    toolbar: {
        toggleSidebar: "Alternar barra lateral",
        support: "Apoyar el proyecto",
        github: "Repositorio GitHub",
        documentation: "Documentación del proyecto",
        import: "Importar pipeline",
        export: "Exportar pipeline",
        getCode: "Obtener código Python",
        toggleTheme: "Cambiar tema",
        runPipeline: "Ejecutar pipeline",
        language: "Idioma"
    },
    sidebar: {
        generation: "Generación",
        sampling: "Muestreo",
        bulkOptics: "Óptica volumétrica",
        photonics: "Fotónica",
        detection: "Detección",
        result: "Resultado",
        running: "Ejecutando simulación...",
        downloadImage: "Descargar imagen"
    },
    elements: {
        scene: "Fuente de escena",
        atmosphere: "Atmósfera",
        telescope: "Telescopio",
        lens: "Lente",
        beamSplitter: "Divisor de haz",
        coronagraph: "Coronógrafo",
        fiberIn: "Entrada de fibra",
        fiberOut: "Salida de fibra",
        mmi: "Acoplador MMI",
        camera: "Cámara"
    },
    validation: {
        selfLoop: "Un nodo no puede conectarse a sí mismo",
        invalidConnection: "Conexión inválida",
        generationLayer: "Las capas de generación (Escena, Atmósfera) solo pueden conectarse a otras capas de generación o capas de muestreo (Telescopio)",
        samplingLayer: "Las capas de muestreo (Telescopio) solo pueden conectarse a capas ópticas (Lente, etc.) o capas de detección (Cámara)",
        opticalLayer: "Las capas ópticas (Lente, Coronógrafo, etc.) solo pueden conectarse a otras capas ópticas o capas de detección (Cámara)",
        detectionLayer: "Las capas de detección (Cámara) solo pueden conectarse a capas de procesamiento de datos",
        dataLayer: "Las capas de datos solo pueden conectarse a otras capas de datos"
    },
    config: {
        focalLength: "Distancia focal",
        splitRatio: "Relación de división",
        maskType: "Tipo de máscara",
        modes: "Modos",
        inputs: "Entradas",
        outputs: "Salidas",
        fourQuadrants: "4 cuadrantes",
        vortex: "Vórtice"
    },
    actions: {
        add: "Añadir",
        remove: "Eliminar",
        close: "Cerrar",
        download: "Descargar",
        copy: "Copiar",
        paste: "Pegar",
        delete: "Eliminar",
        undo: "Deshacer",
        redo: "Rehacer"
    },
    nodes: {
        generation: "Generación",
        sampling: "Muestreo",
        bulkOptics: "Óptica volumétrica",
        photonics: "Fotónica",
        detection: "Detección",
        elements: "Elementos",
        in: "Entrada",
        out: "Salida",
        dropHere: "Suelta elementos aquí para añadirlos...",
        deleteLayer: "Eliminar capa",
        inputConnection: "Conexión de entrada",
        outputConnection: "Conexión de salida"
    }
};
