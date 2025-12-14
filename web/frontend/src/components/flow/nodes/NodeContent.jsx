import React from 'react';
import { Trash2, Eye } from 'lucide-react';
import LayerVisualizer from './LayerVisualizer';
import SignalVisualizer from './SignalVisualizer';

export default function NodeContent({
    id,
    label,
    iconPath,
    Icon,
    config,
    setConfig,
    fields,
    hasInput,
    hasOutput,
    onInspect,
    onDelete,
    selected,
    isDark
}) {
    // Heuristic for capacity: 'modes' for fiber, or 1 default
    const capacity = config.modes || 1;

    const handleChange = (field, value) => {
        const newConfig = { ...config, [field]: value };
        setConfig(newConfig);
    };

    return (
        <div className={`bg-white dark:bg-slate-800 rounded-lg border shadow-xl min-w-[280px] relative transition-colors duration-200 ${selected ? 'border-blue-500 ring-2 ring-blue-500 ring-opacity-50' : 'border-slate-200 dark:border-slate-700'}`}>

            <div className="bg-slate-50 dark:bg-slate-900 px-4 py-3 border-b border-slate-200 dark:border-slate-800 rounded-t-lg flex items-center justify-between">
                <div className="flex items-center gap-2">
                    <div className="p-1.5 bg-cyan-100 dark:bg-cyan-900/30 rounded-md text-cyan-600 dark:text-cyan-400">
                        {iconPath ? (
                            <img src={iconPath} alt={label} className="w-4 h-4 dark:invert dark:opacity-80" />
                        ) : Icon ? (
                            <Icon className="w-4 h-4" />
                        ) : null}
                    </div>
                    <div>
                        <h3 className="font-semibold text-slate-800 dark:text-slate-200 text-sm">{label}</h3>
                        <p className="text-[10px] text-slate-500 uppercase tracking-wider font-bold">Optical Layer</p>
                    </div>
                </div>
                <div className="flex gap-2">
                    <LayerVisualizer type="generic" config={config} />
                    {onInspect && (
                        <button
                            onClick={() => onInspect(id)}
                            className="p-1 px-1.5 rounded hover:bg-slate-200 dark:hover:bg-slate-800 text-slate-400 hover:text-blue-500 transition-colors"
                            title="Inspect Wavefront"
                        >
                            <Eye className="w-3.5 h-3.5" />
                        </button>
                    )}
                    {onDelete && (
                        <button
                            onClick={() => onDelete(id)}
                            className="p-1 px-1.5 rounded hover:bg-slate-200 dark:hover:bg-slate-800 text-slate-400 hover:text-red-500 transition-colors"
                            title="Delete Layer"
                        >
                            <Trash2 className="w-3.5 h-3.5" />
                        </button>
                    )}
                </div>
            </div>

            <div className="p-4 space-y-3 nodrag nowheel text-sm">

                {hasInput && <SignalVisualizer capacity={capacity} />}

                {fields && fields.map((field) => (
                    <div key={field.name} className="flex flex-col gap-1">
                        <label className="text-xs uppercase font-bold text-slate-500">{field.label}</label>
                        {field.type === 'select' ? (
                            <select
                                value={config[field.name]}
                                onChange={(e) => handleChange(field.name, e.target.value)}
                                className={`text-sm rounded px-2 py-1 border outline-none focus:ring-1 focus:ring-blue-500 ${isDark ? 'bg-slate-900 border-slate-700' : 'bg-slate-50 border-slate-300'}`}
                            >
                                {field.options.map(opt => (
                                    <option key={opt.value} value={opt.value}>{opt.label}</option>
                                ))}
                            </select>
                        ) : field.type === 'number' ? (
                            <input
                                type="number"
                                step={field.step || "any"}
                                value={config[field.name]}
                                onChange={(e) => handleChange(field.name, parseFloat(e.target.value))}
                                className={`text-sm rounded px-2 py-1 border outline-none focus:ring-1 focus:ring-blue-500 ${isDark ? 'bg-slate-900 border-slate-700' : 'bg-slate-50 border-slate-300'}`}
                            />
                        ) : (
                            <input
                                type="text"
                                value={config[field.name]}
                                onChange={(e) => handleChange(field.name, e.target.value)}
                                className={`text-sm rounded px-2 py-1 border outline-none focus:ring-1 focus:ring-blue-500 ${isDark ? 'bg-slate-900 border-slate-700' : 'bg-slate-50 border-slate-300'}`}
                            />
                        )}
                    </div>
                ))}
            </div>
        </div>
    );
}
