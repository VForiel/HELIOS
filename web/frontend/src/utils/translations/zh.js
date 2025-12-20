export default {
    app: {
        title: "HELIOS",
        subtitle: "可视化架构师"
    },
    toolbar: {
        toggleSidebar: "切换侧边栏",
        support: "支持项目",
        github: "GitHub仓库",
        documentation: "项目文档",
        import: "导入管道",
        export: "导出管道",
        getCode: "获取Python代码",
        toggleTheme: "切换主题",
        runPipeline: "运行管道",
        language: "语言",
        graphView: "图形视图",
        layeredView: "分层视图",
        autoArrange: "自动排列"
    },
    sidebar: {
        generation: "生成",
        sampling: "采样",
        bulkOptics: "体光学",
        photonics: "光子学",
        detection: "检测",
        result: "结果",
        running: "正在运行模拟...",
        downloadImage: "下载图像"
    },
    elements: {
        scene: "场景源",
        atmosphere: "大气",
        telescope: "望远镜",
        lens: "透镜",
        beamSplitter: "分束器",
        coronagraph: "日冕仪",
        fiberIn: "光纤输入",
        fiberOut: "光纤输出",
        mmi: "MMI耦合器",
        camera: "相机"
    },
    validation: {
        selfLoop: "节点不能连接到自身",
        invalidConnection: "无效连接",
        generationLayer: "生成层（场景、大气）只能连接到其他生成层或采样层（望远镜）",
        samplingLayer: "采样层（望远镜）只能连接到光学层（透镜等）或检测层（相机）",
        opticalLayer: "光学层（透镜、日冕仪等）只能连接到其他光学层或检测层（相机）",
        detectionLayer: "检测层（相机）只能连接到数据处理层",
        dataLayer: "数据层只能连接到其他数据层"
    },
    config: {
        focalLength: "焦距",
        splitRatio: "分束比",
        maskType: "掩模类型",
        modes: "模式",
        inputs: "输入",
        outputs: "输出",
        fourQuadrants: "四象限",
        vortex: "涡旋"
    },
    actions: {
        add: "添加",
        remove: "移除",
        close: "关闭",
        download: "下载",
        copy: "复制",
        paste: "粘贴",
        delete: "删除",
        undo: "撤销",
        redo: "重做"
    },
    nodes: {
        generation: "生成",
        sampling: "采样",
        bulkOptics: "体光学",
        photonics: "光子学",
        detection: "检测",
        elements: "元素",
        in: "输入",
        out: "输出",
        dropHere: "将项目拖放到此处以添加元素...",
        deleteLayer: "删除层",
        inputConnection: "输入连接",
        outputConnection: "输出连接"
    }
};
