import React from 'react';
import { Handle, Position, useReactFlow } from 'reactflow';
import TelescopeConfig from '../../TelescopeConfig';
import LayerVisualizer from './LayerVisualizer';
import { Trash2, Search, Grid } from 'lucide-react'; // Added Grid icon

export default function TelescopeArrayNode({ id, data, selected }) {
    const { deleteElements } = useReactFlow();

    // No need for ResizeObserver anymore as we have a single static handle

    return (
        <div className={`bg-white dark:bg-slate-800 rounded-lg border shadow-xl min-w-[350px] relative transition-colors duration-200 ${selected ? 'border-cyan-500 ring-2 ring-cyan-500 ring-opacity-50' : 'border-slate-200 dark:border-slate-700'}`}>
            <Handle type="target" position={Position.Left} className="!bg-cyan-500 !-left-4 !w-4 !h-4" />

            {/* Layer Header */}
            <div className="bg-slate-50 dark:bg-slate-900 px-4 py-3 border-b border-slate-200 dark:border-slate-800 rounded-t-lg flex items-center justify-between">
                <div className="flex items-center gap-2">
                    <div className="p-1.5 bg-cyan-100 dark:bg-cyan-900/30 rounded-md text-cyan-600 dark:text-cyan-400">
                        <Grid className="w-4 h-4" />
                    </div>
                    <div>
                        <h3 className="font-semibold text-slate-800 dark:text-slate-200 text-sm">Telescope Array</h3>
                        <p className="text-[10px] text-slate-500 uppercase tracking-wider font-bold">Sampling Layer</p>
                    </div>
                </div>

                <div className="flex gap-2">
                    <LayerVisualizer type="telescope" config={data.config} />
                    <button
                        onClick={() => deleteElements({ nodes: [{ id }] })}
                        className="p-1.5 rounded hover:bg-slate-200 dark:hover:bg-slate-800 text-slate-400 hover:text-red-500 transition-colors"
                        title="Delete Layer"
                    >
                        <Trash2 className="w-4 h-4" />
                    </button>
                </div>
            </div>

            {/* Layer Body */}
            <div className="p-0">
                {/* We pass the 'isNode' prop or similar if we want TelescopeConfig to render differently, 
                    but for now we just use the existing config component as the 'Section' content */}
                <div className="max-h-[500px] overflow-y-auto custom-scrollbar p-4 nodrag nowheel">
                    <TelescopeConfig config={data.config} setConfig={data.setConfig} mode="array" />
                </div>
            </div>

            {/* Single Output Handle */}
            <Handle
                type="source"
                position={Position.Right}
                className="!bg-purple-500 !-right-4 !w-4 !h-4"
            />
        </div>
    );
}
