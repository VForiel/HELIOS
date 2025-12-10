import React from 'react';
import { Handle, Position, useReactFlow } from 'reactflow';
import SceneConfig from '../../SceneConfig';
import LayerVisualizer from './LayerVisualizer';
import { Stars, Trash2 } from 'lucide-react';

export default function SceneNode({ id, data, selected }) {
    const { deleteElements } = useReactFlow();

    return (
        <div className={`bg-white dark:bg-slate-800 rounded-lg border shadow-xl min-w-[300px] relative transition-colors duration-200 ${selected ? 'border-blue-500 ring-2 ring-blue-500 ring-opacity-50' : 'border-slate-200 dark:border-slate-700'}`}>
            <div className="bg-slate-50 dark:bg-slate-900 px-4 py-3 border-b border-slate-200 dark:border-slate-800 rounded-t-lg flex items-center justify-between">
                <div className="flex items-center gap-2">
                    <div className="p-1.5 bg-yellow-100 dark:bg-yellow-900/30 rounded-md text-yellow-600 dark:text-yellow-400">
                        <Stars className="w-4 h-4" />
                    </div>
                    <div>
                        <h3 className="font-semibold text-slate-800 dark:text-slate-200 text-sm">Scene / Source</h3>
                        <p className="text-[10px] text-slate-500 uppercase tracking-wider font-bold">Optical Layer</p>
                    </div>
                </div>
                <div className="flex gap-2">
                    <LayerVisualizer type="scene" config={{ stars: data.stars, planets: data.planets, zodiacal: data.zodiacal }} />
                    <button
                        onClick={() => deleteElements({ nodes: [{ id }] })}
                        className="p-1 rounded hover:bg-slate-200 dark:hover:bg-slate-800 text-slate-400 hover:text-red-500 transition-colors"
                        title="Delete Layer"
                    >
                        <Trash2 className="w-4 h-4" />
                    </button>
                </div>
            </div>

            <div className="p-4 max-h-[400px] overflow-y-auto custom-scrollbar nodrag nowheel">
                {/* 'nodrag' class prevents dragging the node when interacting with inputs */}
                <SceneConfig
                    stars={data.stars} setStars={data.setStars}
                    planets={data.planets} setPlanets={data.setPlanets}
                    zodiacal={data.zodiacal} setZodiacal={data.setZodiacal}
                />
            </div>

            <Handle type="source" position={Position.Right} id="source" className="!bg-yellow-500 !-right-4 !w-4 !h-4" />
        </div>
    );
}
