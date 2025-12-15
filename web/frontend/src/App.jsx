import React, { useState } from 'react';
import { Play, Download } from 'lucide-react';
import { ReactFlowProvider, useNodesState, useEdgesState } from 'reactflow';
import PipelineEditor from './components/flow/PipelineEditor';
import LayeredView from './components/LayeredView';
import ErrorBoundary from './components/ErrorBoundary';
import { getElementIcon } from './utils/iconMap';
import { useTranslation } from './utils/i18n';
// Keep config components for passing state logic or if used internally
import SceneConfig from './components/SceneConfig';
import TelescopeConfig from './components/TelescopeConfig';
import AtmosphereConfig from './components/AtmosphereConfig';
import { LayoutList, GitGraph } from 'lucide-react';

function App() {
    // Language state
    const { t, language, setLanguage, languages } = useTranslation();

    // State
    const [sceneConfig, setSceneConfig] = React.useState({
        stars: [{ temperature: 5778, magnitude: 4.83, x_arcsec: 0, y_arcsec: 0 }],
        planets: [{ mass: 1.0, separation: 1.0, angle: 0.0, radius: 1.0 }],
        zodiacal: { enabled: false, brightness: 1.0, radius: null }
    });
    const [atmosphereConfig, setAtmosphereConfig] = React.useState({
        enabled: false,
        rms_nm: 100,
        wind_speed: 5.0
    });
    const [telescopeConfig, setTelescopeConfig] = React.useState({
        preset: 'Single',
        diameter: 8.0,
        pupil_type: 'Circular',
        central_obstruction: 0,
        spiders: 0,
        positions: [{ id: 'single', x: 0, y: 0 }] // Single telescope always at origin for now?
    });
    const [cameraConfig, setCameraConfig] = useState({
        wavelength: 1.0,
        exposure: 0.1
    });

    const [loading, setLoading] = useState(false);
    const [image, setImage] = useState(null);
    const [error, setError] = useState(null);

    // UI State
    const [isSidebarOpen, setIsSidebarOpen] = useState(false);
    const [theme, setTheme] = useState('dark');
    const [viewMode, setViewMode] = useState('graph'); // 'graph' | 'layered'

    // Hoisted Graph State
    // Initial Nodes (Layers)
    const initialNodes = [
        {
            id: 'layer-1',
            type: 'layer',
            position: { x: 50, y: 100 },
            data: {
                elements: [
                    { type: 'scene', label: 'Scene', config: sceneConfig, iconPath: getElementIcon('scene') }
                ]
            }
        },
        {
            id: 'layer-2',
            type: 'layer',
            position: { x: 500, y: 100 },
            data: {
                elements: [
                    { type: 'telescope', label: 'Telescope', config: telescopeConfig, iconPath: getElementIcon('telescope') }
                ]
            }
        },
        {
            id: 'layer-3',
            type: 'layer',
            position: { x: 950, y: 100 },
            data: {
                elements: [
                    { type: 'camera', label: 'Camera', config: cameraConfig, iconPath: getElementIcon('camera') }
                ]
            }
        }
    ];

    const initialEdges = [
        { id: 'e1-2', source: 'layer-1', target: 'layer-2', animated: true },
        { id: 'e2-3', source: 'layer-2', target: 'layer-3', animated: true },
    ];

    const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
    const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);

    // Theme Effect
    React.useEffect(() => {
        if (theme === 'dark') {
            document.documentElement.classList.add('dark');
        } else {
            document.documentElement.classList.remove('dark');
        }
    }, [theme]);

    const toggleTheme = () => setTheme(prev => prev === 'dark' ? 'light' : 'dark');

    // --- Pipeline Execution ---
    const runPipeline = async (pipeline_layers) => {
        setLoading(true);
        setError(null);
        try {
            // Transform pipeline object to backend expected format
            // IF backend supports dynamic pipeline, send list.
            // ELSE, map to fixed structure.

            // For now, we are going to UPDATE the backend to support pipeline or use legacy.
            // Let's first map the linear pipeline back to the 'fixed' config needed by current backend 
            // OR update backend. 
            // The plan said: "Endpoint Update... Refactor /simulate"
            // So we will assume we send the whole pipeline list or a new struct.

            const payload = {
                mode: 'pipeline',
                layers: pipeline_layers
            };

            // NOTE: The current backend expects SimulationConfig (fixed fields).
            // We will need to update backend next. For now, frontend prepares the logic.

            const response = await fetch('/api/simulate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });

            if (!response.ok) {
                const errText = await response.text();
                throw new Error(`Simulation failed: ${response.status} - ${errText}`);
            }

            const blob = await response.blob();
            const imageUrl = URL.createObjectURL(blob);
            setImage(imageUrl);
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    const onDragStart = (event, nodeType) => {
        event.dataTransfer.setData('application/reactflow', nodeType);
        event.dataTransfer.effectAllowed = 'move';
    };

    return (
        <ErrorBoundary>
            <ReactFlowProvider>
                <div className={`flex h-screen font-sans overflow-hidden transition-colors duration-300 ${theme === 'dark' ? 'bg-slate-900 text-slate-100' : 'bg-slate-50 text-slate-900'}`}>

                    {/* TOOLBOX SIDEBAR */}
                    <div className={`${isSidebarOpen ? 'w-64 translate-x-0' : 'w-0 -translate-x-full opacity-0'} transition-all duration-300 ease-in-out flex-shrink-0 bg-white dark:bg-slate-900 border-r border-slate-200 dark:border-slate-800 flex flex-col h-full z-20 shadow-xl overflow-hidden`}>
                        {/* Sidebar Header removed as it is now in Top Bar */}
                        <div className="p-4 space-y-2 min-w-[256px] mt-4 overflow-y-auto custom-scrollbar flex-1 pb-20">

                            {/* Generation */}
                            <div className="text-xs font-bold text-blue-500 uppercase tracking-wider mb-2">{t('sidebar.generation')}</div>
                            <div className="bg-slate-50 dark:bg-slate-800 p-3 rounded cursor-grab border border-slate-200 dark:border-slate-700 hover:border-blue-500 transition-colors flex items-center"
                                onDragStart={(event) => onDragStart(event, 'scene')} draggable>
                                <img src={getElementIcon('scene')} alt="Scene" className="w-4 h-4 mr-3 dark:invert dark:opacity-80" />
                                <span className="text-slate-700 dark:text-slate-200 text-sm">{t('elements.scene')}</span>
                            </div>
                            <div className="bg-slate-50 dark:bg-slate-800 p-3 rounded cursor-grab border border-slate-200 dark:border-slate-700 hover:border-blue-500 transition-colors flex items-center"
                                onDragStart={(event) => onDragStart(event, 'atmosphere')} draggable>
                                <img src={getElementIcon('atmosphere')} alt="Atmosphere" className="w-4 h-4 mr-3 dark:invert dark:opacity-80" />
                                <span className="text-slate-700 dark:text-slate-200 text-sm">{t('elements.atmosphere')}</span>
                            </div>

                            {/* Sampling */}
                            <div className="text-xs font-bold text-cyan-500 uppercase tracking-wider mb-2 mt-6">{t('sidebar.sampling')}</div>
                            <div className="bg-slate-50 dark:bg-slate-800 p-3 rounded cursor-grab border border-slate-200 dark:border-slate-700 hover:border-cyan-500 transition-colors flex items-center mb-2"
                                onDragStart={(event) => onDragStart(event, 'telescope')} draggable>
                                <img src={getElementIcon('telescope')} alt="Telescope" className="w-4 h-4 mr-3 dark:invert dark:opacity-80" />
                                <span className="text-slate-700 dark:text-slate-200 text-sm">{t('elements.telescope')}</span>
                            </div>
                            <div className="bg-slate-50 dark:bg-slate-800 p-3 rounded cursor-grab border border-slate-200 dark:border-slate-700 hover:border-cyan-500 transition-colors flex items-center"
                                onDragStart={(event) => onDragStart(event, 'telescope_array')} draggable>
                                <img src={getElementIcon('telescope')} alt="Telescope Array" className="w-4 h-4 mr-3 dark:invert dark:opacity-80" />
                                <span className="text-slate-700 dark:text-slate-200 text-sm">Telescope Array</span>
                            </div>

                            {/* Bulk Optics */}
                            <div className="text-xs font-bold text-indigo-500 uppercase tracking-wider mb-2 mt-6">{t('sidebar.bulkOptics')}</div>
                            <div className="bg-slate-50 dark:bg-slate-800 p-3 rounded cursor-grab border border-slate-200 dark:border-slate-700 hover:border-indigo-500 transition-colors flex items-center"
                                onDragStart={(event) => onDragStart(event, 'lens')} draggable>
                                <img src={getElementIcon('lens')} alt="Lens" className="w-4 h-4 mr-3 dark:invert dark:opacity-80" />
                                <span className="text-slate-700 dark:text-slate-200 text-sm">{t('elements.lens')}</span>
                            </div>
                            <div className="bg-slate-50 dark:bg-slate-800 p-3 rounded cursor-grab border border-slate-200 dark:border-slate-700 hover:border-indigo-500 transition-colors flex items-center"
                                onDragStart={(event) => onDragStart(event, 'beam_splitter')} draggable>
                                <img src={getElementIcon('beam_splitter')} alt="Beam Splitter" className="w-4 h-4 mr-3 dark:invert dark:opacity-80" />
                                <span className="text-slate-700 dark:text-slate-200 text-sm">{t('elements.beamSplitter')}</span>
                            </div>
                            <div className="bg-slate-50 dark:bg-slate-800 p-3 rounded cursor-grab border border-slate-200 dark:border-slate-700 hover:border-indigo-500 transition-colors flex items-center"
                                onDragStart={(event) => onDragStart(event, 'coronagraph')} draggable>
                                <img src={getElementIcon('coronagraph')} alt="Coronagraph" className="w-4 h-4 mr-3 dark:invert dark:opacity-80" />
                                <span className="text-slate-700 dark:text-slate-200 text-sm">{t('elements.coronagraph')}</span>
                            </div>

                            {/* Photonics */}
                            <div className="text-xs font-bold text-amber-500 uppercase tracking-wider mb-2 mt-6">{t('sidebar.photonics')}</div>
                            <div className="bg-slate-50 dark:bg-slate-800 p-3 rounded cursor-grab border border-slate-200 dark:border-slate-700 hover:border-amber-500 transition-colors flex items-center"
                                onDragStart={(event) => onDragStart(event, 'fiber_in')} draggable>
                                <img src={getElementIcon('fiber_in')} alt="Fiber Input" className="w-4 h-4 mr-3 dark:invert dark:opacity-80" />
                                <span className="text-slate-700 dark:text-slate-200 text-sm">{t('elements.fiberIn')}</span>
                            </div>
                            <div className="bg-slate-50 dark:bg-slate-800 p-3 rounded cursor-grab border border-slate-200 dark:border-slate-700 hover:border-amber-500 transition-colors flex items-center"
                                onDragStart={(event) => onDragStart(event, 'fiber_out')} draggable>
                                <img src={getElementIcon('fiber_out')} alt="Fiber Output" className="w-4 h-4 mr-3 dark:invert dark:opacity-80" />
                                <span className="text-slate-700 dark:text-slate-200 text-sm">{t('elements.fiberOut')}</span>
                            </div>
                            <div className="bg-slate-50 dark:bg-slate-800 p-3 rounded cursor-grab border border-slate-200 dark:border-slate-700 hover:border-amber-500 transition-colors flex items-center"
                                onDragStart={(event) => onDragStart(event, 'mmi')} draggable>
                                <img src={getElementIcon('mmi')} alt="MMI Coupler" className="w-4 h-4 mr-3 dark:invert dark:opacity-80" />
                                <span className="text-slate-700 dark:text-slate-200 text-sm">{t('elements.mmi')}</span>
                            </div>

                            {/* Detection */}
                            <div className="text-xs font-bold text-pink-500 uppercase tracking-wider mb-2 mt-6">{t('sidebar.detection')}</div>
                            <div className="bg-slate-50 dark:bg-slate-800 p-3 rounded cursor-grab border border-slate-200 dark:border-slate-700 hover:border-pink-500 transition-colors flex items-center"
                                onDragStart={(event) => onDragStart(event, 'camera')} draggable>
                                <img src={getElementIcon('camera')} alt="Camera" className="w-4 h-4 mr-3 dark:invert dark:opacity-80" />
                                <span className="text-slate-700 dark:text-slate-200 text-sm">{t('elements.camera')}</span>
                            </div>
                        </div>

                        <div className="mt-auto p-4 border-t border-slate-200 dark:border-slate-800 min-w-[256px]">
                            {image && (
                                <div className="relative group cursor-pointer" onClick={() => window.open(image, '_blank')}>
                                    <div className="absolute top-0 right-0 bg-blue-600 text-xs px-2 py-1 rounded-bl text-white z-10">{t('sidebar.result')}</div>
                                    <div className="absolute bottom-2 right-2 bg-white/10 backdrop-blur-sm p-1.5 rounded hover:bg-white/20 transition-colors z-10"
                                        onClick={(e) => {
                                            e.stopPropagation();
                                            const a = document.createElement('a');
                                            a.href = image;
                                            a.download = "simulation_result.png";
                                            document.body.appendChild(a);
                                            a.click();
                                            setTimeout(() => document.body.removeChild(a), 1000);
                                        }}
                                        title={t('sidebar.downloadImage')}
                                    >
                                        <Download className="w-4 h-4 text-white" />
                                    </div>
                                    <img src={image} className="w-full h-32 object-contain bg-black rounded border border-slate-300 dark:border-slate-700" alt="Result thumbnail" />
                                </div>
                            )}
                            {loading && <div className="text-center text-xs text-blue-500 dark:text-blue-400 mt-2 animate-pulse">{t('sidebar.running')}</div>}
                            {error && <div className="text-xs text-red-500 dark:text-red-400 mt-2 bg-red-100 dark:bg-red-900/20 p-2 rounded">{error}</div>}
                        </div>
                    </div>

                    {/* GRAPH EDITOR AREA */}
                    <div className="flex-1 relative h-full bg-slate-50 dark:bg-slate-950">
                        <PipelineEditor
                            nodes={nodes}
                            setNodes={setNodes}
                            onNodesChange={onNodesChange}
                            edges={edges}
                            setEdges={setEdges}
                            onEdgesChange={onEdgesChange}


                            stars={sceneConfig.stars}
                            setStars={(v) => setSceneConfig(prev => ({ ...prev, stars: typeof v === 'function' ? v(prev.stars) : v }))}
                            planets={sceneConfig.planets}
                            setPlanets={(v) => setSceneConfig(prev => ({ ...prev, planets: typeof v === 'function' ? v(prev.planets) : v }))}
                            zodiacal={sceneConfig.zodiacal}
                            setZodiacal={(v) => setSceneConfig(prev => ({ ...prev, zodiacal: typeof v === 'function' ? v(prev.zodiacal) : v }))}

                            atmosphere={atmosphereConfig} setAtmosphere={setAtmosphereConfig}
                            telescope={telescopeConfig} setTelescope={setTelescopeConfig}
                            camera={cameraConfig} setCamera={setCameraConfig}

                            runSimulation={runPipeline}
                            onToggleSidebar={() => setIsSidebarOpen(!isSidebarOpen)}
                            onToggleTheme={toggleTheme}
                            isDark={theme === 'dark'}
                            language={language}
                            setLanguage={setLanguage}
                            languages={languages}
                            t={t}
                            viewMode={viewMode}
                            setViewMode={setViewMode}
                        />
                    </div>
                </div>
            </ReactFlowProvider>
        </ErrorBoundary>
    );
}

export default App;
