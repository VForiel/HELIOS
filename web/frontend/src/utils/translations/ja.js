export default {
    app: {
        title: "HELIOS",
        subtitle: "ビジュアルアーキテクト"
    },
    toolbar: {
        toggleSidebar: "サイドバーを切り替え",
        support: "プロジェクトを支援",
        github: "GitHubリポジトリ",
        documentation: "プロジェクトドキュメント",
        import: "パイプラインをインポート",
        export: "パイプラインをエクスポート",
        getCode: "Pythonコードを取得",
        toggleTheme: "テーマを切り替え",
        runPipeline: "パイプラインを実行",
        language: "言語",
        graphView: "グラフビュー",
        layeredView: "レイヤービュー",
        autoArrange: "自動配置"
    },
    sidebar: {
        generation: "生成",
        sampling: "サンプリング",
        bulkOptics: "バルク光学",
        photonics: "フォトニクス",
        detection: "検出",
        result: "結果",
        running: "シミュレーション実行中...",
        downloadImage: "画像をダウンロード"
    },
    elements: {
        scene: "シーンソース",
        atmosphere: "大気",
        telescope: "望遠鏡",
        lens: "レンズ",
        beamSplitter: "ビームスプリッター",
        coronagraph: "コロナグラフ",
        fiberIn: "ファイバー入力",
        fiberOut: "ファイバー出力",
        mmi: "MMIカプラ",
        camera: "カメラ"
    },
    validation: {
        selfLoop: "ノードは自分自身に接続できません",
        invalidConnection: "無効な接続",
        generationLayer: "生成レイヤー（シーン、大気）は、他の生成レイヤーまたはサンプリングレイヤー（望遠鏡）にのみ接続できます",
        samplingLayer: "サンプリングレイヤー（望遠鏡）は、光学レイヤー（レンズなど）または検出レイヤー（カメラ）にのみ接続できます",
        opticalLayer: "光学レイヤー（レンズ、コロナグラフなど）は、他の光学レイヤーまたは検出レイヤー（カメラ）にのみ接続できます",
        detectionLayer: "検出レイヤー（カメラ）は、データ処理レイヤーにのみ接続できます",
        dataLayer: "データレイヤーは、他のデータレイヤーにのみ接続できます"
    },
    config: {
        focalLength: "焦点距離",
        splitRatio: "分割比",
        maskType: "マスクタイプ",
        modes: "モード",
        inputs: "入力",
        outputs: "出力",
        fourQuadrants: "4象限",
        vortex: "渦"
    },
    actions: {
        add: "追加",
        remove: "削除",
        close: "閉じる",
        download: "ダウンロード",
        copy: "コピー",
        paste: "貼り付け",
        delete: "削除",
        undo: "元に戻す",
        redo: "やり直す"
    },
    nodes: {
        generation: "生成",
        sampling: "サンプリング",
        bulkOptics: "バルク光学",
        photonics: "フォトニクス",
        detection: "検出",
        elements: "要素",
        in: "入力",
        out: "出力",
        dropHere: "要素を追加するにはここにドロップ...",
        deleteLayer: "レイヤーを削除",
        inputConnection: "入力接続",
        outputConnection: "出力接続"
    }
};
