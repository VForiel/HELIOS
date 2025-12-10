import React, { useState } from 'react';
import { Trash2, ChevronDown, ChevronRight, Eye, Settings } from 'lucide-react';
import LayerVisualizer from './LayerVisualizer';

// Import Config Components
import SceneConfig from '../../SceneConfig';
import TelescopeConfig from '../../TelescopeConfig';

// Icons need to be passed or mapped. 
// For simplicity, we might pass the Icon component in the element definition or map it here.

export default function ElementRow({ element, index, onChange, onRemove }) {
    const [expanded, setExpanded] = useState(false);

    // Unpack element data
    const { type, config, label, icon: IconComponent } = element;

    // Helper to update config
    const setConfig = (newConfig) => {
        onChange(index, { ...element, config: newConfig });
    };

    // Specific Updaters for complex components (Scene/Telescope) that expect specific props
    const setStars = (v) => setConfig({ ...config, stars: v });
    const setPlanets = (v) => setConfig({ ...config, planets: v });
    const setZodiacal = (v) => setConfig({ ...config, zodiacal: v });

    return (
        <div className="border border-slate-200 dark:border-slate-700 rounded-md bg-white dark:bg-slate-900/50 mb-2 overflow-hidden">
            {/* Header / Summary Line */}
            <div className="flex items-center justify-between p-2 bg-slate-50 dark:bg-slate-800/50 hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors">
                <div
                    className="flex items-center gap-2 flex-1 cursor-pointer"
                    onClick={() => setExpanded(!expanded)}
                >
                    {expanded ? <ChevronDown className="w-3 h-3 text-slate-400" /> : <ChevronRight className="w-3 h-3 text-slate-400" />}

                    <div className="p-1 rounded text-slate-500 bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 shadow-sm">
                        {IconComponent ? <IconComponent className="w-3.5 h-3.5" /> : <Settings className="w-3.5 h-3.5" />}
                    </div>

                    <span className="text-xs font-semibold text-slate-700 dark:text-slate-300 select-none">
                        {label || type}
                    </span>
                </div>

                <div className="flex items-center gap-1">
                    {/* Visualizer is specific to Elements now */}
                    <LayerVisualizer type={type} config={config} />

                    <button
                        onClick={() => onRemove(index)}
                        className="p-1 text-slate-400 hover:text-red-500 hover:bg-slate-200 dark:hover:bg-slate-700 rounded transition-colors"
                        title="Remove Element"
                    >
                        <Trash2 className="w-3 h-3" />
                    </button>
                </div>
            </div>

            {/* Expanded Config Body */}
            {expanded && (
                <div className="p-3 border-t border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 text-sm">
                    {type === 'scene' ? (
                        <SceneConfig
                            stars={config.stars} setStars={setStars}
                            planets={config.planets} setPlanets={setPlanets}
                            zodiacal={config.zodiacal} setZodiacal={setZodiacal}
                        />
                    ) : type === 'telescope' ? (
                        <TelescopeConfig config={config} setConfig={setConfig} />
                    ) : (
                        /* Generic Config Rendering */
                        <div className="space-y-2">
                            {/* Specific mapping for known types or generic field loop */}
                            {element.fields ? (
                                element.fields.map(field => (
                                    <div key={field.name}>
                                        <label className="block text-xs font-medium text-slate-500 mb-1">{field.label}</label>
                                        {field.type === 'select' ? (
                                            <select
                                                value={config[field.name]}
                                                onChange={(e) => setConfig({ ...config, [field.name]: field.type === 'number' ? parseFloat(e.target.value) : e.target.value })}
                                                className="w-full text-xs p-1.5 rounded border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-800 text-slate-900 dark:text-slate-100"
                                            >
                                                {field.options.map(opt => <option key={opt.value} value={opt.value}>{opt.label}</option>)}
                                            </select>
                                        ) : (
                                            <input
                                                type={field.type}
                                                step={field.step}
                                                value={config[field.name] || ''}
                                                onChange={(e) => setConfig({ ...config, [field.name]: field.type === 'number' ? parseFloat(e.target.value) : e.target.value })}
                                                className="w-full text-xs p-1.5 rounded border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-800 text-slate-900 dark:text-slate-100 font-mono"
                                            />
                                        )}
                                    </div>
                                ))
                            ) : (
                                /* Fallback for Camera or unknown types without fields def */
                                type === 'camera' ? (
                                    <>
                                        <div>
                                            <label className="block text-slate-500 mb-1 text-xs">Exposure (s)</label>
                                            <input type="number" value={config.exposure || 0.1} onChange={(e) => setConfig({ ...config, exposure: parseFloat(e.target.value) })} className="w-full border rounded px-2 py-1 text-xs dark:bg-slate-800 dark:border-slate-700" />
                                        </div>
                                        <div className="mt-2">
                                            <label className="block text-slate-500 mb-1 text-xs">Wavelength (um)</label>
                                            <input type="number" value={config.wavelength || 1.0} onChange={(e) => setConfig({ ...config, wavelength: parseFloat(e.target.value) })} className="w-full border rounded px-2 py-1 text-xs dark:bg-slate-800 dark:border-slate-700" />
                                        </div>
                                    </>
                                ) : <div className="text-xs text-slate-400 italic">No configuration available.</div>
                            )}
                        </div>
                    )}
                </div>
            )}
        </div>
    );
}
