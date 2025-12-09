import React from 'react';
import { Handle, Position, useReactFlow } from 'reactflow';
import { Trash2, Settings } from 'lucide-react';

export default function GenericNode({ id, data, selected }) {
    const { deleteElements } = useReactFlow();

    const handleChange = (field, value) => {
        const newConfig = { ...data.config, [field]: value };
        data.setConfig(newConfig);
    };

    return (
        <div className={`rounded-lg border shadow-xl min-w-[200px] relative transition-all ${data.isDark ? 'bg-slate-800' : 'bg-white'} ${selected ? 'border-blue-500 ring-2 ring-blue-500 ring-opacity-50' : (data.isDark ? 'border-slate-700' : 'border-slate-200')}`}>
            {/* Header */}
            <div className={`px-4 py-2 border-b rounded-t-lg font-semibold flex items-center justify-between ${data.isDark ? 'bg-slate-900 border-slate-800' : 'bg-slate-50 border-slate-200'} ${data.colorClass || 'text-blue-500'}`}>
                <div className="flex items-center">
                    {data.icon && <data.icon className="w-4 h-4 mr-2" />}
                    {data.label || 'Component'}
                </div>
                <button
                    onClick={() => deleteElements({ nodes: [{ id }] })}
                    className="p-1 rounded hover:opacity-80 transition-opacity text-slate-500 hover:text-red-500"
                    title="Delete Node"
                >
                    <Trash2 className="w-4 h-4" />
                </button>
            </div>

            {/* Body */}
            <div className={`p-4 space-y-3 nodrag ${data.isDark ? 'text-slate-200' : 'text-slate-700'}`}>
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
