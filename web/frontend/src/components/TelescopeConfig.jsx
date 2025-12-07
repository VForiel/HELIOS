import React from 'react';
import { Eye, Settings, Plus, Trash2 } from 'lucide-react';

export default function TelescopeConfig({ config, setConfig }) {

    const [selectedPresetToLoad, setSelectedPresetToLoad] = React.useState('VLTI-UT');
    const [isLoading, setIsLoading] = React.useState(false);

    const loadPreset = async () => {
        setIsLoading(true);
        try {
            const response = await fetch(`/api/presets/${selectedPresetToLoad}`);
            if (response.ok) {
                const collectorsData = await response.json();
                // Inject unique IDs for React Flow handles
                const newCollectors = collectorsData.map(col => ({
                    ...col,
                    id: crypto.randomUUID()
                }));

                // Replace current collectors with loaded ones, keep preset as 'Custom'
                setConfig({ ...config, preset: 'Custom', collectors: newCollectors });
            } else {
                console.error("Failed to load preset");
            }
        } catch (e) {
            console.error(e);
        } finally {
            setIsLoading(false);
        }
    };

    const addCollector = () => {
        setConfig({
            ...config,
            collectors: [...config.collectors, {
                id: crypto.randomUUID(),
                x: 0,
                y: 0,
                diameter: 8.0,
                pupil_type: 'Circular',
                central_obstruction: 0,
                spiders: 0
            }]
        });
    };

    const updateCollector = (index, field, value) => {
        const newCols = [...config.collectors];
        // Handle number parsing
        if (['x', 'y', 'diameter', 'central_obstruction', 'spiders'].includes(field)) {
            newCols[index][field] = parseFloat(value);
        } else {
            newCols[index][field] = value;
        }
        setConfig({ ...config, collectors: newCols });
    };

    const removeCollector = (index) => {
        const newCols = config.collectors.filter((_, i) => i !== index);
        setConfig({ ...config, collectors: newCols });
    };

    return (
        <div className="bg-white dark:bg-slate-800 p-5 rounded-lg border border-slate-200 dark:border-slate-700 shadow-sm dark:shadow-none">
            <h3 className="text-lg font-medium text-purple-600 dark:text-purple-400 mb-4 flex items-center">
                <Eye className="w-5 h-5 mr-2" /> Configuration
            </h3>

            {/* Preset Loader */}
            <div className="mb-6 p-3 bg-slate-50 dark:bg-slate-900/50 rounded border border-slate-200 dark:border-slate-800">
                <label className="block text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2">Load Preset</label>
                <div className="flex gap-2">
                    <select
                        value={selectedPresetToLoad}
                        onChange={(e) => setSelectedPresetToLoad(e.target.value)}
                        className="flex-1 bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded px-3 py-1.5 text-sm text-slate-900 dark:text-slate-100 focus:outline-none focus:border-purple-500"
                    >
                        <option value="VLTI-UT">VLTI (4 x 8.2m UTs)</option>
                        <option value="VLTI-AT">VLTI (4 x 1.8m ATs)</option>
                        <option value="LIFE">LIFE Mission (Formation Flying)</option>
                    </select>
                    <button
                        onClick={loadPreset}
                        disabled={isLoading}
                        className="px-4 py-1.5 bg-purple-600 hover:bg-purple-700 text-white text-sm rounded transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                        {isLoading ? 'Loading...' : 'Load'}
                    </button>
                </div>
                <p className="text-xs text-slate-400 mt-2">Loading a preset will replace current collectors.</p>
            </div>

            {/* Collector List - Always Visible */}
            <div className="space-y-3">

                <p className="text-xs text-slate-500">Define collectors for interferometry or sparse aperture.</p>

                {config.collectors.map((col, index) => (
                    <div key={index} className="bg-slate-50 dark:bg-slate-900/50 p-3 rounded border border-slate-200 dark:border-slate-800 text-sm">
                        <div className="flex justify-between items-center mb-2">
                            <span className="font-semibold text-slate-700 dark:text-slate-300">Collector #{index + 1}</span>
                            <button onClick={() => removeCollector(index)} className="text-red-400 hover:text-red-300">
                                <Trash2 className="w-3.5 h-3.5" />
                            </button>
                        </div>
                        <div className="grid grid-cols-2 gap-2">
                            <div>
                                <label className="text-xs text-slate-500 mb-1">X (m)</label>
                                <input type="number" value={col.x} onChange={(e) => updateCollector(index, 'x', e.target.value)}
                                    className="w-full bg-white dark:bg-slate-800 rounded px-2 py-1 border border-slate-300 dark:border-slate-700 font-mono text-xs text-slate-900 dark:text-slate-100" />
                            </div>
                            <div>
                                <label className="text-xs text-slate-500 mb-1">Y (m)</label>
                                <input type="number" value={col.y} onChange={(e) => updateCollector(index, 'y', e.target.value)}
                                    className="w-full bg-white dark:bg-slate-800 rounded px-2 py-1 border border-slate-300 dark:border-slate-700 font-mono text-xs text-slate-900 dark:text-slate-100" />
                            </div>
                            <div className="col-span-2">
                                <label className="text-xs text-slate-500 mb-1">Diameter (m)</label>
                                <input type="number" value={col.diameter} onChange={(e) => updateCollector(index, 'diameter', e.target.value)}
                                    className="w-full bg-white dark:bg-slate-800 rounded px-2 py-1 border border-slate-300 dark:border-slate-700 font-mono text-xs text-slate-900 dark:text-slate-100" />
                            </div>

                            <div className="col-span-2">
                                <label className="text-xs text-slate-500 mb-1">Pupil Type</label>
                                <select value={col.pupil_type} onChange={(e) => updateCollector(index, 'pupil_type', e.target.value)}
                                    className="w-full bg-white dark:bg-slate-800 rounded px-2 py-1 border border-slate-300 dark:border-slate-700 text-xs text-slate-900 dark:text-slate-100">
                                    <option value="Circular">Circular</option>
                                    <option value="Obstructed">Obstructed</option>
                                    <option value="VLT">VLT-like</option>
                                    <option value="JWST">JWST-like</option>
                                </select>
                            </div>

                            {col.pupil_type === 'Obstructed' && (
                                <>
                                    <div>
                                        <label className="text-xs text-slate-500 mb-1">Obstruction (0-1)</label>
                                        <input type="number" step="0.05" value={col.central_obstruction} onChange={(e) => updateCollector(index, 'central_obstruction', e.target.value)}
                                            className="w-full bg-white dark:bg-slate-800 rounded px-2 py-1 border border-slate-300 dark:border-slate-700 font-mono text-xs text-slate-900 dark:text-slate-100" />
                                    </div>
                                    <div>
                                        <label className="text-xs text-slate-500 mb-1">Spider Arms</label>
                                        <input type="number" step="1" value={col.spiders} onChange={(e) => updateCollector(index, 'spiders', e.target.value)}
                                            className="w-full bg-white dark:bg-slate-800 rounded px-2 py-1 border border-slate-300 dark:border-slate-700 font-mono text-xs text-slate-900 dark:text-slate-100" />
                                    </div>
                                </>
                            )}
                        </div>
                    </div>
                ))}
                <button onClick={addCollector} className="w-full py-1.5 rounded border border-dashed border-slate-600 text-slate-400 hover:text-white text-xs flex items-center justify-center">
                    <Plus className="w-3.5 h-3.5 mr-1" /> Add Collector
                </button>
            </div>

        </div>
    );
}
