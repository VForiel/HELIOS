import React from 'react';
import { Eye, Settings, Plus, Trash2 } from 'lucide-react';

export default function TelescopeConfig({ config, setConfig }) {

    const updatePreset = (preset) => {
        // Reset or set defaults based on preset
        if (preset === 'Single') {
            setConfig({ ...config, preset, diameter: 8.0, collectors: [] });
        } else if (preset === 'Custom') {
            setConfig({ ...config, preset, collectors: config.collectors.length ? config.collectors : [{ x: 0, y: 0, diameter: 8.0, pupil_type: 'Circular', central_obstruction: 0, spiders: 0 }] });
        } else {
            setConfig({ ...config, preset });
        }
    };

    const addCollector = () => {
        setConfig({
            ...config,
            collectors: [...config.collectors, { x: 0, y: 0, diameter: 8.0, pupil_type: 'Circular', central_obstruction: 0, spiders: 0 }]
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
        <div className="bg-slate-800 p-5 rounded-lg border border-slate-700">
            <h3 className="text-lg font-medium text-purple-400 mb-4 flex items-center">
                <Eye className="w-5 h-5 mr-2" /> Telescope Array
            </h3>

            <div className="mb-4">
                <label className="block text-sm text-slate-400 mb-2">Preset Configuration</label>
                <select
                    value={config.preset}
                    onChange={(e) => updatePreset(e.target.value)}
                    className="w-full bg-slate-900 border border-slate-700 rounded px-3 py-2 text-sm focus:outline-none focus:border-purple-500"
                >
                    <option value="Single">Single Telescope</option>
                    <option value="VLTI-UT">VLTI (4 x 8.2m UTs)</option>
                    <option value="VLTI-AT">VLTI (4 x 1.8m ATs)</option>
                    <option value="LIFE">LIFE Mission (Formation Flying)</option>
                    <option value="Custom">Custom Array</option>
                </select>
            </div>

            {/* Single Telescope simple config */}
            {config.preset === 'Single' && (
                <div>
                    <label className="block text-sm text-slate-400 mb-1">Diameter (m)</label>
                    <input
                        type="number"
                        value={config.diameter || 8.0}
                        onChange={(e) => setConfig({ ...config, diameter: parseFloat(e.target.value) })}
                        className="w-full bg-slate-900 border border-slate-700 rounded px-3 py-2 text-sm focus:outline-none focus:border-purple-500"
                    />
                </div>
            )}

            {/* Custom Array Config */}
            {config.preset === 'Custom' && (
                <div className="space-y-3">
                    <p className="text-xs text-slate-500">Define collectors for interferometry or sparse aperture.</p>

                    {config.collectors.map((col, index) => (
                        <div key={index} className="bg-slate-900/50 p-3 rounded border border-slate-800 text-sm">
                            <div className="flex justify-between items-center mb-2">
                                <span className="font-semibold text-slate-300">Collector #{index + 1}</span>
                                <button onClick={() => removeCollector(index)} className="text-red-400 hover:text-red-300">
                                    <Trash2 className="w-3.5 h-3.5" />
                                </button>
                            </div>
                            <div className="grid grid-cols-2 gap-2">
                                <div>
                                    <label className="text-xs text-slate-500">X (m)</label>
                                    <input type="number" value={col.x} onChange={(e) => updateCollector(index, 'x', e.target.value)}
                                        className="w-full bg-slate-800 rounded px-2 py-1 border border-slate-700 font-mono text-xs" />
                                </div>
                                <div>
                                    <label className="text-xs text-slate-500">Y (m)</label>
                                    <input type="number" value={col.y} onChange={(e) => updateCollector(index, 'y', e.target.value)}
                                        className="w-full bg-slate-800 rounded px-2 py-1 border border-slate-700 font-mono text-xs" />
                                </div>
                                <div className="col-span-2">
                                    <label className="text-xs text-slate-500">Diameter (m)</label>
                                    <input type="number" value={col.diameter} onChange={(e) => updateCollector(index, 'diameter', e.target.value)}
                                        className="w-full bg-slate-800 rounded px-2 py-1 border border-slate-700 font-mono text-xs" />
                                </div>

                                <div className="col-span-2">
                                    <label className="text-xs text-slate-500">Pupil Type</label>
                                    <select value={col.pupil_type} onChange={(e) => updateCollector(index, 'pupil_type', e.target.value)}
                                        className="w-full bg-slate-800 rounded px-2 py-1 border border-slate-700 text-xs">
                                        <option value="Circular">Circular</option>
                                        <option value="Obstructed">Obstructed</option>
                                        <option value="VLT">VLT-like</option>
                                        <option value="JWST">JWST-like</option>
                                    </select>
                                </div>

                                {col.pupil_type === 'Obstructed' && (
                                    <>
                                        <div>
                                            <label className="text-xs text-slate-500">Obstruction (0-1)</label>
                                            <input type="number" step="0.05" value={col.central_obstruction} onChange={(e) => updateCollector(index, 'central_obstruction', e.target.value)}
                                                className="w-full bg-slate-800 rounded px-2 py-1 border border-slate-700 font-mono text-xs" />
                                        </div>
                                        <div>
                                            <label className="text-xs text-slate-500">Spider Arms</label>
                                            <input type="number" step="1" value={col.spiders} onChange={(e) => updateCollector(index, 'spiders', e.target.value)}
                                                className="w-full bg-slate-800 rounded px-2 py-1 border border-slate-700 font-mono text-xs" />
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
            )}
        </div>
    );
}
