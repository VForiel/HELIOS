import React from 'react';
import { Handle, Position, useReactFlow, useUpdateNodeInternals } from 'reactflow';
import TelescopeConfig from '../../TelescopeConfig';
import LayerVisualizer from './LayerVisualizer';
import { Trash2 } from 'lucide-react';

export default function TelescopeNode({ id, data, selected }) {
    const { deleteElements } = useReactFlow();
    const updateNodeInternals = useUpdateNodeInternals();

    // Determine outputs: 1 per collector, or 1 default if empty
    const collectors = data.config.collectors || [];

    // Force update handles when collectors change
    const nodeRef = React.useRef(null);
    const prevHeight = React.useRef(0);

    // Update handles whenever node size triggers a resize (e.g. adding inputs)
    React.useEffect(() => {
        if (!nodeRef.current) return;

        const observer = new ResizeObserver(() => {
            // Check if height actually changed to avoid loop
            if (nodeRef.current && Math.abs(nodeRef.current.offsetHeight - prevHeight.current) > 1) {
                prevHeight.current = nodeRef.current.offsetHeight;
                // Add immediate buffer + slight delay for safety
                updateNodeInternals(id);
                setTimeout(() => updateNodeInternals(id), 50);
            }
        });

        observer.observe(nodeRef.current);
        return () => observer.disconnect();
    }, [id, updateNodeInternals]);

    // Also force update on data change (collectors count)
    React.useEffect(() => {
        const t = setTimeout(() => updateNodeInternals(id), 20);
        return () => clearTimeout(t);
    }, [id, updateNodeInternals, collectors.length]);

    return (
        <div ref={nodeRef} className={`bg-white dark:bg-slate-800 rounded-lg border shadow-xl min-w-[350px] relative transition-colors duration-200 ${selected ? 'border-blue-500 ring-2 ring-blue-500 ring-opacity-50' : 'border-slate-200 dark:border-slate-700'}`}>
            <Handle type="target" position={Position.Left} className="!bg-cyan-500 !-left-4 !w-4 !h-4" />

            <div className="bg-slate-50 dark:bg-slate-900 px-4 py-2 border-b border-slate-200 dark:border-slate-800 rounded-t-lg font-semibold text-purple-600 dark:text-purple-400 flex items-center justify-between">
                <div className="flex items-center">
                    Telescopes
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

            {/* Dynamic Output Handles - One per collector */}
            {collectors.length > 0 ? (
                collectors.map((col, index) => (
                    <Handle
                        key={col.id || index}
                        type="source"
                        position={Position.Right}
                        id={col.id || `out-${index}`}
                        style={{ top: `${((index + 1) * 100) / (collectors.length + 1)}%` }}
                        className="!bg-purple-500 !-right-4 !w-4 !h-4"
                        title={`Collector ${index + 1}`}
                    />
                ))
            ) : (
                /* Fallback Default Handle if no collectors defined */
                <Handle
                    type="source"
                    position={Position.Right}
                    id="out-default"
                    className="!bg-purple-500 !-right-4 !w-4 !h-4"
                    title="Output"
                />
            )}

            <div className="p-4 max-h-[400px] overflow-y-auto custom-scrollbar nodrag nowheel">
                <TelescopeConfig config={data.config} setConfig={data.setConfig} />
            </div>
        </div>
    );
}
