import React from 'react';
import { Handle, Position, useReactFlow } from 'reactflow';
import { Trash2 } from 'lucide-react';
import LayerVisualizer from './LayerVisualizer';
import SignalVisualizer from './SignalVisualizer';

export default function GenericNode({ id, data, selected }) {
    const { deleteElements } = useReactFlow();
    const config = data.config || {};
    const Icon = data.icon;

    // Heuristic for capacity: 'modes' for fiber, or 1 default
    const capacity = config.modes || 1;

    const handleChange = (field, value) => {
        const newConfig = { ...data.config, [field]: value };
        data.setConfig(newConfig);
    };

    return (
        <div className={`bg-white dark:bg-slate-800 rounded-lg border shadow-xl min-w-[280px] relative transition-colors duration-200 ${selected ? 'border-blue-500 ring-2 ring-blue-500 ring-opacity-50' : 'border-slate-200 dark:border-slate-700'}`}>
            {data.hasInput && <Handle type="target" position={Position.Left} className="!bg-blue-500 !-left-4 !w-4 !h-4" />}

            <div className="bg-slate-50 dark:bg-slate-900 px-4 py-3 border-b border-slate-200 dark:border-slate-800 rounded-t-lg flex items-center justify-between">
                <div className="flex items-center gap-2">
                    <div className="p-1.5 bg-cyan-100 dark:bg-cyan-900/30 rounded-md text-cyan-600 dark:text-cyan-400">
                        {Icon && <Icon className="w-4 h-4" />}
                    </div>
                    <div>
                        <h3 className="font-semibold text-slate-800 dark:text-slate-200 text-sm">{data.label}</h3>
                        <p className="text-[10px] text-slate-500 uppercase tracking-wider font-bold">Optical Layer</p>
                    </div>
                </div>
                <div className="flex gap-2">
                    <LayerVisualizer type="generic" config={config} />
                    <button
                        onClick={() => deleteElements({ nodes: [{ id }] })}
                        className="p-1 px-1.5 rounded hover:bg-slate-200 dark:hover:bg-slate-800 text-slate-400 hover:text-red-500 transition-colors"
                        title="Delete Layer"
                    >
                        <Trash2 className="w-3.5 h-3.5" />
                    </button>
                </div>
            </div>

            <div className="p-4 space-y-3 nodrag nowheel text-sm">

                {data.hasInput && <SignalVisualizer capacity={capacity} />}

                {data.fields && data.fields.map((field) => (
                    <div key={field.name} className="flex flex-col gap-1">
                        <label className="text-xs uppercase font-bold text-slate-500">{field.label}</label>
                        {field.type === 'select' ? (
                            <select
                                value={data.config[field.name]}
                                onChange={(e) => handleChange(field.name, e.target.value)}
                                className={`text-sm rounded px-2 py-1 border outline-none focus:ring-1 focus:ring-blue-500 ${data.isDark ? 'bg-slate-900 border-slate-700' : 'bg-slate-50 border-slate-300'}`}
                            >
                                {field.options.map(opt => (
                                    <option key={opt.value} value={opt.value}>{opt.label}</option>
                                ))}
                            </select>
                        ) : field.type === 'number' ? (
                            <input
                                type="number"
                                step={field.step || "any"}
                                value={data.config[field.name]}
                                onChange={(e) => handleChange(field.name, parseFloat(e.target.value))}
                                className={`text-sm rounded px-2 py-1 border outline-none focus:ring-1 focus:ring-blue-500 ${data.isDark ? 'bg-slate-900 border-slate-700' : 'bg-slate-50 border-slate-300'}`}
                            />
                        ) : (
                            <input
                                type="text"
                                value={data.config[field.name]}
                                onChange={(e) => handleChange(field.name, e.target.value)}
                                className={`text-sm rounded px-2 py-1 border outline-none focus:ring-1 focus:ring-blue-500 ${data.isDark ? 'bg-slate-900 border-slate-700' : 'bg-slate-50 border-slate-300'}`}
                            />
                        )}
                    </div>
                ))}
            </div>

            {/* Ports */}
            {/* Input Port (Left) - Generic */}
            {data.hasInput !== false && (
                <Handle
                    type="target"
                    position={Position.Left}
                    id="in"
                    className={`!w-3 !h-3 !-left-3 ${data.handleClass || '!bg-slate-400'}`}
                />
            )}

            {/* Output Port (Right) - Generic */}
            {data.hasOutput !== false && (
                <Handle
                    type="source"
                    position={Position.Right}
                    id="out"
                    className={`!w-3 !h-3 !-right-3 ${data.handleClass || '!bg-slate-400'}`}
                />
            )}
        </div>
    );
}
