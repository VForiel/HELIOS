import React from 'react';
import { Handle, Position, useReactFlow } from 'reactflow';
import { Camera, Trash2 } from 'lucide-react';

import LayerVisualizer from './LayerVisualizer';

export default function CameraNode({ id, data, selected }) {
    const { deleteElements } = useReactFlow();

    return (
        <div className={`bg-white dark:bg-slate-800 rounded-lg border shadow-xl min-w-[250px] relative transition-colors duration-200 ${selected ? 'border-blue-500 ring-2 ring-blue-500 ring-opacity-50' : 'border-slate-200 dark:border-slate-700'}`}>
            <Handle type="target" id="target" position={Position.Left} className="!bg-purple-500 !-left-4 !w-4 !h-4" />
            <div className="bg-slate-50 dark:bg-slate-900 px-4 py-2 border-b border-slate-200 dark:border-slate-800 rounded-t-lg font-semibold text-pink-600 dark:text-pink-400 flex items-center justify-between">
                <div className="flex items-center">
                    <Camera className="w-4 h-4 mr-2" /> Camera / Detector
                </div>
                <div className="flex gap-2">
                    <LayerVisualizer type="camera" config={data.config} />
                    <button
                        onClick={() => deleteElements({ nodes: [{ id }] })}
                        className="p-1 rounded hover:bg-slate-200 dark:hover:bg-slate-700 text-slate-500 dark:text-slate-400 hover:text-red-500 dark:hover:text-red-400 transition-colors"
                        title="Delete Node"
                    >
                        <Trash2 className="w-4 h-4" />
                    </button>
                </div>
            </div>
            <div className="p-4 space-y-3 nodrag text-sm">
                <div>
                    <label className="block text-slate-500 mb-1">Pixels</label>
                    <div className="flex gap-2">
                        <input type="number" value={256} disabled className="w-full bg-slate-100 dark:bg-slate-900 rounded px-2 py-1 border border-slate-200 dark:border-slate-700 opacity-50 cursor-not-allowed text-slate-500 dark:text-slate-400" />
                        <span className="text-slate-500 self-center">x</span>
                        <input type="number" value={256} disabled className="w-full bg-slate-100 dark:bg-slate-900 rounded px-2 py-1 border border-slate-200 dark:border-slate-700 opacity-50 cursor-not-allowed text-slate-500 dark:text-slate-400" />
                    </div>
                </div>
                <div>
                    <label className="block text-slate-500 mb-1">Exposure (s)</label>
                    <input
                        type="number"
                        value={data.config.exposure || 0.1}
                        onChange={(e) => data.setConfig({ ...data.config, exposure: parseFloat(e.target.value) })}
                        className="w-full bg-white dark:bg-slate-900 rounded px-2 py-1 border border-slate-300 dark:border-slate-700 text-slate-900 dark:text-slate-100 focus:outline-none focus:border-pink-500"
                    />
                </div>
                <div>
                    <label className="block text-slate-500 mb-1">Wavelength (um)</label>
                    <input
                        type="number"
                        value={data.config.wavelength || 1.0}
                        onChange={(e) => data.setConfig({ ...data.config, wavelength: parseFloat(e.target.value) })}
                        className="w-full bg-white dark:bg-slate-900 rounded px-2 py-1 border border-slate-300 dark:border-slate-700 text-slate-900 dark:text-slate-100 focus:outline-none focus:border-pink-500"
                    />
                </div>
            </div>
        </div>
    );
}
