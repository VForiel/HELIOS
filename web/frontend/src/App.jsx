import React, { useState } from 'react';
import { Play, Download } from 'lucide-react';
import { ReactFlowProvider } from 'reactflow';
import PipelineEditor from './components/flow/PipelineEditor';
import ErrorBoundary from './components/ErrorBoundary';
// Keep config components for passing state logic or if used internally
import SceneConfig from './components/SceneConfig';
import TelescopeConfig from './components/TelescopeConfig';
import AtmosphereConfig from './components/AtmosphereConfig';

function App() {
    // State
    const [stars, setStars] = useState([{ temperature: 5778, magnitude: 4.83, x_arcsec: 0, y_arcsec: 0 }]);
    const [planets, setPlanets] = useState([{ mass: 1.0, separation: 1.0, angle: 0.0, radius: 1.0 }]);
    const [zodiacal, setZodiacal] = useState({ enabled: false, brightness: 1.0, radius: null });

    const [atmosphere, setAtmosphere] = useState({ enabled: false, rms_nm: 100, wind_speed: 5.0 });

    const [telescope, setTelescope] = useState({
        preset: 'Custom',
        diameter: 8.0,
        collectors: [{ x: 0, y: 0, diameter: 8.0, pupil_type: 'Circular', central_obstruction: 0, spiders: 0 }]
    });
    const [camera, setCamera] = useState({ wavelength: 1.0, exposure: 0.1 });

    const [loading, setLoading] = useState(false);
    const [image, setImage] = useState(null);
    const [error, setError] = useState(null);

    // UI State
    const [isSidebarOpen, setIsSidebarOpen] = useState(false);
    const [theme, setTheme] = useState('dark');

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
                        <div className="p-4 space-y-4 min-w-[256px] mt-4">
                            <div className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-2">Block Library</div>

                            <div className="bg-slate-50 dark:bg-slate-800 p-3 rounded cursor-grab border border-slate-200 dark:border-slate-700 hover:border-blue-500 transition-colors flex items-center"
                                onDragStart={(event) => onDragStart(event, 'scene')} draggable>
                                <div className="w-3 h-3 rounded-full bg-blue-500 mr-3"></div>
                                <span className="text-slate-700 dark:text-slate-200">Scene Source</span>
                            </div>

                            <div className="bg-slate-50 dark:bg-slate-800 p-3 rounded cursor-grab border border-slate-200 dark:border-slate-700 hover:border-cyan-500 transition-colors flex items-center"
                                onDragStart={(event) => onDragStart(event, 'atmosphere')} draggable>
                                <div className="w-3 h-3 rounded-full bg-cyan-500 mr-3"></div>
                                <span className="text-slate-700 dark:text-slate-200">Atmosphere Layer</span>
                            </div>

                            <div className="bg-slate-50 dark:bg-slate-800 p-3 rounded cursor-grab border border-slate-200 dark:border-slate-700 hover:border-purple-500 transition-colors flex items-center"
                                onDragStart={(event) => onDragStart(event, 'telescope')} draggable>
                                <div className="w-3 h-3 rounded-full bg-purple-500 mr-3"></div>
                                <span className="text-slate-700 dark:text-slate-200">Telescope</span>
                            </div>

                            <div className="bg-slate-50 dark:bg-slate-800 p-3 rounded cursor-grab border border-slate-200 dark:border-slate-700 hover:border-pink-500 transition-colors flex items-center"
                                onDragStart={(event) => onDragStart(event, 'camera')} draggable>
                                <div className="w-3 h-3 rounded-full bg-pink-500 mr-3"></div>
                                <span className="text-slate-700 dark:text-slate-200">Camera / Detector</span>
                            </div>
                        </div>

                        <div className="mt-auto p-4 border-t border-slate-200 dark:border-slate-800 min-w-[256px]">
                            {image && (
                                <div className="relative group cursor-pointer" onClick={() => window.open(image, '_blank')}>
                                    <div className="absolute top-0 right-0 bg-blue-600 text-xs px-2 py-1 rounded-bl text-white z-10">Result</div>
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
                                        title="Download Image"
                                    >
                                        <Download className="w-4 h-4 text-white" />
                                    </div>
                                    <img src={image} className="w-full h-32 object-contain bg-black rounded border border-slate-300 dark:border-slate-700" alt="Result thumbnail" />
                                </div>
                            )}
                            {loading && <div className="text-center text-xs text-blue-500 dark:text-blue-400 mt-2 animate-pulse">Running Simulation...</div>}
                            {error && <div className="text-xs text-red-500 dark:text-red-400 mt-2 bg-red-100 dark:bg-red-900/20 p-2 rounded">{error}</div>}
                        </div>
                    </div>

                    {/* GRAPH EDITOR AREA */}
                    <div className="flex-1 relative h-full bg-slate-50 dark:bg-slate-950">
                        <PipelineEditor
                            stars={stars} setStars={setStars}
                            planets={planets} setPlanets={setPlanets}
                            zodiacal={zodiacal} setZodiacal={setZodiacal}
                            atmosphere={atmosphere} setAtmosphere={setAtmosphere}
                            telescope={telescope} setTelescope={setTelescope}
                            camera={camera} setCamera={setImage} // Basic camera node handling
                            runSimulation={runPipeline}
                            onToggleSidebar={() => setIsSidebarOpen(!isSidebarOpen)}
                            onToggleTheme={toggleTheme}
                            isDark={theme === 'dark'}
                        />
                    </div>
                </div>
            </ReactFlowProvider>
        </ErrorBoundary>
    );
}

export default App;
