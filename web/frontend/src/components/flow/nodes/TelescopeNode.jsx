import React from 'react';
import { Handle, Position, useReactFlow } from 'reactflow';
import TelescopeConfig from '../../TelescopeConfig';
import LayerVisualizer from './LayerVisualizer';
import { Trash2 } from 'lucide-react';

export default function TelescopeNode({ id, data }) {
    const { deleteElements } = useReactFlow();

    // Determine outputs: 1 per collector, or 1 default if empty
    const collectors = data.config.collectors || [];
    const outputCount = collectors.length > 0 ? collectors.length : 1;

    return (
        <div className="bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700 shadow-xl min-w-[350px] relative">
            <Handle type="target" position={Position.Left} className="!bg-cyan-500 !-left-3 !w-3 !h-3" />

            <div className="bg-slate-50 dark:bg-slate-900 px-4 py-2 border-b border-slate-200 dark:border-slate-800 rounded-t-lg font-semibold text-purple-600 dark:text-purple-400 flex items-center justify-between">
                <div className="flex items-center">
                    Telescope
                </div>
                <div className="flex gap-2">
                    <LayerVisualizer type="telescope" config={data.config} />
                    <button
                        onClick={() => deleteElements({ nodes: [{ id }] })}
                        className="p-1 rounded hover:bg-slate-200 dark:hover:bg-slate-700 text-slate-500 dark:text-slate-400 hover:text-red-500 dark:hover:text-red-400 transition-colors"
                        title="Delete Node"
                    >
                        <Trash2 className="w-4 h-4" />
                    </button>
                </div>
            </div>

            {/* Dynamic Output Handles */}
            {collectors.length > 0 ? (
                collectors.map((_, index) => (
                    <Handle
                        key={index}
                        type="source"
                        position={Position.Right}
                        id={`out-${index}`}
                        style={{ top: `${((index + 1) * 100) / (outputCount + 1)}%` }}
                        className="!bg-purple-500 !-right-3 !w-3 !h-3"
                        title={`Collector ${index + 1}`}
                    />
                ))
            ) : (
                <Handle
                    type="source"
                    position={Position.Right}
                    className="!bg-purple-500 !-right-3 !w-3 !h-3"
                    title="Array Output"
                />
            )}

            <div className="p-4 max-h-[400px] overflow-y-auto custom-scrollbar nodrag nowheel">
                <TelescopeConfig config={data.config} setConfig={data.setConfig} />
            </div>
        </div>
    );
}
