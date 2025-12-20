export default {
    app: {
        title: "HELIOS",
        subtitle: "Architecte Visuel"
    },
    toolbar: {
        toggleSidebar: "Basculer la barre latérale",
        support: "Soutenir le projet",
        github: "Dépôt GitHub",
        documentation: "Documentation du projet",
        import: "Importer un pipeline",
        export: "Exporter le pipeline",
        getCode: "Obtenir le code Python",
        toggleTheme: "Changer le thème",
        runPipeline: "Exécuter le pipeline",
        language: "Langue",
        graphView: "Vue graphe",
        layeredView: "Vue en couches",
        autoArrange: "Réorganiser automatiquement"
    },
    sidebar: {
        generation: "Génération",
        sampling: "Échantillonnage",
        bulkOptics: "Optique volumique",
        photonics: "Photonique",
        detection: "Détection",
        result: "Résultat",
        running: "Simulation en cours...",
        downloadImage: "Télécharger l'image"
    },
    elements: {
        scene: "Source de scène",
        atmosphere: "Atmosphère",
        telescope: "Télescope",
        lens: "Lentille",
        beamSplitter: "Séparateur de faisceau",
        coronagraph: "Coronographe",
        fiberIn: "Entrée de fibre",
        fiberOut: "Sortie de fibre",
        mmi: "Coupleur MMI",
        camera: "Caméra"
    },
    validation: {
        selfLoop: "Un nœud ne peut pas se connecter à lui-même",
        invalidConnection: "Connexion invalide",
        generationLayer: "Les couches de génération (Scène, Atmosphère) peuvent seulement se connecter à d'autres couches de génération ou à des couches d'échantillonnage (Télescope)",
        samplingLayer: "Les couches d'échantillonnage (Télescope) peuvent seulement se connecter à des couches optiques (Lentille, etc.) ou de détection (Caméra)",
        opticalLayer: "Les couches optiques (Lentille, Coronographe, etc.) peuvent seulement se connecter à d'autres couches optiques ou de détection (Caméra)",
        detectionLayer: "Les couches de détection (Caméra) peuvent seulement se connecter à des couches de traitement de données",
        dataLayer: "Les couches de données peuvent seulement se connecter à d'autres couches de données"
    },
    config: {
        focalLength: "Distance focale",
        splitRatio: "Rapport de division",
        maskType: "Type de masque",
        modes: "Modes",
        inputs: "Entrées",
        outputs: "Sorties",
        fourQuadrants: "4 quadrants",
        vortex: "Vortex"
    },
    actions: {
        add: "Ajouter",
        remove: "Supprimer",
        close: "Fermer",
        download: "Télécharger",
        copy: "Copier",
        paste: "Coller",
        delete: "Supprimer",
        undo: "Annuler",
        redo: "Rétablir"
    },
    nodes: {
        generation: "Génération",
        sampling: "Échantillonnage",
        bulkOptics: "Optique volumique",
        photonics: "Photonique",
        detection: "Détection",
        elements: "Éléments",
        in: "Entrée",
        out: "Sortie",
        dropHere: "Déposez des éléments ici pour les ajouter...",
        deleteLayer: "Supprimer la couche",
        inputConnection: "Connexion d'entrée",
        outputConnection: "Connexion de sortie"
    }
};
