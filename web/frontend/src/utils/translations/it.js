export default {
    app: {
        title: "HELIOS",
        subtitle: "Architetto Visuale"
    },
    toolbar: {
        toggleSidebar: "Attiva/disattiva barra laterale",
        support: "Sostieni il progetto",
        github: "Repository GitHub",
        documentation: "Documentazione del progetto",
        import: "Importa pipeline",
        export: "Esporta pipeline",
        getCode: "Ottieni codice Python",
        toggleTheme: "Cambia tema",
        runPipeline: "Esegui pipeline",
        language: "Lingua"
    },
    sidebar: {
        generation: "Generazione",
        sampling: "Campionamento",
        bulkOptics: "Ottica volumetrica",
        photonics: "Fotonica",
        detection: "Rilevamento",
        result: "Risultato",
        running: "Esecuzione simulazione...",
        downloadImage: "Scarica immagine"
    },
    elements: {
        scene: "Sorgente scena",
        atmosphere: "Atmosfera",
        telescope: "Telescopio",
        lens: "Lente",
        beamSplitter: "Divisore di fascio",
        coronagraph: "Coronografo",
        fiberIn: "Ingresso fibra",
        fiberOut: "Uscita fibra",
        mmi: "Accoppiatore MMI",
        camera: "Fotocamera"
    },
    validation: {
        selfLoop: "Un nodo non può connettersi a se stesso",
        invalidConnection: "Connessione non valida",
        generationLayer: "I livelli di generazione (Scena, Atmosfera) possono connettersi solo ad altri livelli di generazione o livelli di campionamento (Telescopio)",
        samplingLayer: "I livelli di campionamento (Telescopio) possono connettersi solo a livelli ottici (Lente, ecc.) o livelli di rilevamento (Fotocamera)",
        opticalLayer: "I livelli ottici (Lente, Coronografo, ecc.) possono connettersi solo ad altri livelli ottici o livelli di rilevamento (Fotocamera)",
        detectionLayer: "I livelli di rilevamento (Fotocamera) possono connettersi solo a livelli di elaborazione dati",
        dataLayer: "I livelli dati possono connettersi solo ad altri livelli dati"
    },
    config: {
        focalLength: "Lunghezza focale",
        splitRatio: "Rapporto di divisione",
        maskType: "Tipo di maschera",
        modes: "Modi",
        inputs: "Ingressi",
        outputs: "Uscite",
        fourQuadrants: "4 quadranti",
        vortex: "Vortice"
    },
    actions: {
        add: "Aggiungi",
        remove: "Rimuovi",
        close: "Chiudi",
        download: "Scarica",
        copy: "Copia",
        paste: "Incolla",
        delete: "Elimina",
        undo: "Annulla",
        redo: "Ripeti"
    },
    nodes: {
        generation: "Generazione",
        sampling: "Campionamento",
        bulkOptics: "Ottica volumetrica",
        photonics: "Fotonica",
        detection: "Rilevamento",
        elements: "Elementi",
        in: "Ingresso",
        out: "Uscita",
        dropHere: "Trascina elementi qui per aggiungerli...",
        deleteLayer: "Elimina livello",
        inputConnection: "Connessione di ingresso",
        outputConnection: "Connessione di uscita"
    }
};
