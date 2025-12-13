export default {
    app: {
        title: "HELIOS",
        subtitle: "Visueller Architekt"
    },
    toolbar: {
        toggleSidebar: "Seitenleiste umschalten",
        support: "Projekt unterstützen",
        github: "GitHub-Repository",
        documentation: "Projektdokumentation",
        import: "Pipeline importieren",
        export: "Pipeline exportieren",
        getCode: "Python-Code abrufen",
        toggleTheme: "Design wechseln",
        runPipeline: "Pipeline ausführen",
        language: "Sprache"
    },
    sidebar: {
        generation: "Erzeugung",
        sampling: "Abtastung",
        bulkOptics: "Volumenoptik",
        photonics: "Photonik",
        detection: "Erkennung",
        result: "Ergebnis",
        running: "Simulation läuft...",
        downloadImage: "Bild herunterladen"
    },
    elements: {
        scene: "Szenenquelle",
        atmosphere: "Atmosphäre",
        telescope: "Teleskop",
        lens: "Linse",
        beamSplitter: "Strahlteiler",
        coronagraph: "Koronograph",
        fiberIn: "Fasereingang",
        fiberOut: "Faserausgang",
        mmi: "MMI-Koppler",
        camera: "Kamera"
    },
    validation: {
        selfLoop: "Ein Knoten kann nicht mit sich selbst verbunden werden",
        invalidConnection: "Ungültige Verbindung",
        generationLayer: "Erzeugungsschichten (Szene, Atmosphäre) können nur mit anderen Erzeugungsschichten oder Abtastschichten (Teleskop) verbunden werden",
        samplingLayer: "Abtastschichten (Teleskop) können nur mit optischen Schichten (Linse usw.) oder Erkennungsschichten (Kamera) verbunden werden",
        opticalLayer: "Optische Schichten (Linse, Koronograph usw.) können nur mit anderen optischen Schichten oder Erkennungsschichten (Kamera) verbunden werden",
        detectionLayer: "Erkennungsschichten (Kamera) können nur mit Datenverarbeitungsschichten verbunden werden",
        dataLayer: "Datenschichten können nur mit anderen Datenschichten verbunden werden"
    },
    config: {
        focalLength: "Brennweite",
        splitRatio: "Teilungsverhältnis",
        maskType: "Maskentyp",
        modes: "Modi",
        inputs: "Eingänge",
        outputs: "Ausgänge",
        fourQuadrants: "4 Quadranten",
        vortex: "Wirbel"
    },
    actions: {
        add: "Hinzufügen",
        remove: "Entfernen",
        close: "Schließen",
        download: "Herunterladen",
        copy: "Kopieren",
        paste: "Einfügen",
        delete: "Löschen",
        undo: "Rückgängig",
        redo: "Wiederholen"
    },
    nodes: {
        generation: "Erzeugung",
        sampling: "Abtastung",
        bulkOptics: "Volumenoptik",
        photonics: "Photonik",
        detection: "Erkennung",
        elements: "Elemente",
        in: "Eingang",
        out: "Ausgang",
        dropHere: "Elemente hier ablegen, um sie hinzuzufügen...",
        deleteLayer: "Ebene löschen",
        inputConnection: "Eingangsverbindung",
        outputConnection: "Ausgangsverbindung"
    }
};
