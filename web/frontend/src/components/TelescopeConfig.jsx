import React from 'react';
import { Eye, Settings, Plus, Trash2 } from 'lucide-react';

export default function TelescopeConfig({ config, setConfig, mode = 'array' }) {
    // Mode can be 'single' or 'array'

    const [selectedPresetToLoad, setSelectedPresetToLoad] = React.useState('VLTI-UT');
    const [isLoading, setIsLoading] = React.useState(false);

    const loadPreset = async () => {
        setIsLoading(true);
        try {
            const response = await fetch(`/api/presets/${selectedPresetToLoad}`);
            if (response.ok) {
                const positionsData = await response.json();
                // Inject unique IDs for React Flow handles
                const newPositions = positionsData.map(pos => ({
                    ...pos,
                    id: crypto.randomUUID()
                }));

                // Replace current positions with loaded ones, keep preset as 'Custom'
                setConfig({ ...config, preset: 'Custom', positions: newPositions });
            } else {
                console.error("Failed to load preset");
            }
        } catch (e) {
            console.error(e);
        } finally {
            setIsLoading(false);
        }
    };

    const addPosition = () => {
        setConfig({
            ...config,
            positions: [...config.positions, {
                id: crypto.randomUUID(),
                x: 0,
                y: 0
            }]
        });
    };

    const updatePosition = (index, field, value) => {
        const newPositions = [...config.positions];
        newPositions[index][field] = parseFloat(value);
        setConfig({ ...config, positions: newPositions });
    };

    const removePosition = (index) => {
        const newPositions = config.positions.filter((_, i) => i !== index);
        setConfig({ ...config, positions: newPositions });
    };

    const updatePupilConfig = (field, value) => {
        if (['diameter', 'central_obstruction', 'spiders'].includes(field)) {
            setConfig({ ...config, [field]: parseFloat(value) });
        } else {
            setConfig({ ...config, [field]: value });
        }
    };

    return (
        <div className="bg-white dark:bg-slate-800 p-5 rounded-lg border border-slate-200 dark:border-slate-700 shadow-sm dark:shadow-none">
            <h3 className="text-lg font-medium text-purple-600 dark:text-purple-400 mb-4 flex items-center">
                <Eye className="w-5 h-5 mr-2" /> Configuration
            </h3>

            {/* Preset Loader - ONLY FOR ARRAY */}
            {mode === 'array' && (
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
                    <p className="text-xs text-slate-400 mt-2">Loading a preset will replace current positions.</p>
                </div>
            )}

            {/* Shared Pupil Configuration - FOR BOTH */}
            <div className="mb-6 p-4 bg-purple-50 dark:bg-purple-900/20 rounded border border-purple-200 dark:border-purple-800">
                <h4 className="text-sm font-semibold text-purple-700 dark:text-purple-300 mb-3 flex items-center">
                    <Settings className="w-4 h-4 mr-2" /> {mode === 'array' ? 'Shared Pupil Configuration' : 'Pupil Configuration'}
                </h4>
                <p className="text-xs text-slate-500 mb-3">
                    {mode === 'array' ? 'All telescopes in the array share the same pupil geometry.' : 'Define the aperture geometry.'}
                </p>

                <div className="grid grid-cols-2 gap-3">
                    <div className="col-span-2">
                        <label className="text-xs text-slate-600 dark:text-slate-400 mb-1 block">Diameter (m)</label>
                        <input
                            type="number"
                            value={config.diameter}
                            onChange={(e) => updatePupilConfig('diameter', e.target.value)}
                            className="w-full bg-white dark:bg-slate-800 rounded px-3 py-2 border border-slate-300 dark:border-slate-700 font-mono text-sm text-slate-900 dark:text-slate-100"
                        />
                    </div>

                    <div className="col-span-2">
                        <label className="text-xs text-slate-600 dark:text-slate-400 mb-1 block">Pupil Type</label>
                        <select
                            value={config.pupil_type}
                            onChange={(e) => updatePupilConfig('pupil_type', e.target.value)}
                            className="w-full bg-white dark:bg-slate-800 rounded px-3 py-2 border border-slate-300 dark:border-slate-700 text-sm text-slate-900 dark:text-slate-100"
                        >
                            <option value="Circular">Circular</option>
                            <option value="Obstructed">Obstructed</option>
                            <option value="VLT">VLT-like</option>
                            <option value="JWST">JWST-like</option>
                        </select>
                    </div>

                    {config.pupil_type === 'Obstructed' && (
                        <>
                            <div>
                                <label className="text-xs text-slate-600 dark:text-slate-400 mb-1 block">Obstruction (0-1)</label>
                                <input
                                    type="number"
                                    step="0.05"
                                    value={config.central_obstruction}
                                    onChange={(e) => updatePupilConfig('central_obstruction', e.target.value)}
                                    className="w-full bg-white dark:bg-slate-800 rounded px-3 py-2 border border-slate-300 dark:border-slate-700 font-mono text-sm text-slate-900 dark:text-slate-100"
                                />
                            </div>
                            <div>
                                <label className="text-xs text-slate-600 dark:text-slate-400 mb-1 block">Spider Arms</label>
                                <input
                                    type="number"
                                    step="1"
                                    value={config.spiders}
                                    onChange={(e) => updatePupilConfig('spiders', e.target.value)}
                                    className="w-full bg-white dark:bg-slate-800 rounded px-3 py-2 border border-slate-300 dark:border-slate-700 font-mono text-sm text-slate-900 dark:text-slate-100"
                                />
                            </div>
                        </>
                    )}
                </div>
            </div>

            {/* Telescope Positions List - ONLY FOR ARRAY */}
            {mode === 'array' && (
                <div className="space-y-3">
                    <h4 className="text-sm font-semibold text-slate-700 dark:text-slate-300">Telescope Positions</h4>
                    <p className="text-xs text-slate-500">Define positions for interferometry or sparse aperture. All telescopes use the shared pupil above.</p>

                    {config.positions.map((pos, index) => (
                        <div key={pos.id || index} className="bg-slate-50 dark:bg-slate-900/50 p-3 rounded border border-slate-200 dark:border-slate-800 text-sm">
                            <div className="flex justify-between items-center mb-2">
                                <span className="font-semibold text-slate-700 dark:text-slate-300">Position #{index + 1}</span>
                                <button
                                    onClick={() => removePosition(index)}
                                    disabled={config.positions.length <= 1}
                                    className={`text-red-400 hover:text-red-300 ${config.positions.length <= 1 ? 'opacity-30 cursor-not-allowed' : ''}`}
                                >
                                    <Trash2 className="w-3.5 h-3.5" />
                                </button>
                            </div>
                            <div className="grid grid-cols-2 gap-2">
                                <div>
                                    <label className="text-xs text-slate-500 mb-1 block">X (m)</label>
                                    <input
                                        type="number"
                                        value={pos.x}
                                        onChange={(e) => updatePosition(index, 'x', e.target.value)}
                                        className="w-full bg-white dark:bg-slate-800 rounded px-2 py-1 border border-slate-300 dark:border-slate-700 font-mono text-xs text-slate-900 dark:text-slate-100"
                                    />
                                </div>
                                <div>
                                    <label className="text-xs text-slate-500 mb-1 block">Y (m)</label>
                                    <input
                                        type="number"
                                        value={pos.y}
                                        onChange={(e) => updatePosition(index, 'y', e.target.value)}
                                        className="w-full bg-white dark:bg-slate-800 rounded px-2 py-1 border border-slate-300 dark:border-slate-700 font-mono text-xs text-slate-900 dark:text-slate-100"
                                    />
                                </div>
                            </div>
                        </div>
                    ))}
                    <button
                        onClick={addPosition}
                        className="w-full py-1.5 rounded border border-dashed border-slate-600 text-slate-400 hover:text-white text-xs flex items-center justify-center"
                    >
                        <Plus className="w-3.5 h-3.5 mr-1" /> Add Position
                    </button>
                </div>
            )}

        </div>
    );
}
