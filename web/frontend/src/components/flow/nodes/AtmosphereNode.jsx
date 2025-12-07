import React from 'react';
import { Handle, Position, useReactFlow } from 'reactflow';
import AtmosphereConfig from '../../AtmosphereConfig';
import LayerVisualizer from './LayerVisualizer';
import { Trash2 } from 'lucide-react';

export default function AtmosphereNode({ id, data }) {
    const { deleteElements } = useReactFlow();

    return (
        <div className="bg-slate-800 rounded-lg border border-slate-700 shadow-xl min-w-[300px] relative">
            <Handle type="target" position={Position.Left} className="!bg-blue-500 !-left-3 !w-3 !h-3" />
            <div className="bg-slate-900 px-4 py-2 border-b border-slate-800 rounded-t-lg font-semibold text-cyan-400 flex items-center justify-between">
                <div className="flex items-center">
                    Atmosphere
                </div>
                <div className="flex gap-2">
                    <LayerVisualizer type="atmosphere" config={data.config} />
                    <button
                        onClick={() => deleteElements({ nodes: [{ id }] })}
                        className="p-1 rounded hover:bg-slate-700 text-slate-400 hover:text-red-400 transition-colors"
                        title="Delete Node"
                    >
                        <Trash2 className="w-4 h-4" />
                    </button>
                </div>
            </div>
            <Handle type="source" position={Position.Right} className="!bg-cyan-500 !-right-3 !w-3 !h-3" />
            <div className="p-4 nodrag">
                <AtmosphereConfig config={data.config} setConfig={data.setConfig} />
            </div>
        </div>
    );
}
