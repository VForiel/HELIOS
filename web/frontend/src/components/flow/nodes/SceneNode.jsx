import React from 'react';
import { Handle, Position, useReactFlow } from 'reactflow';
import SceneConfig from '../../SceneConfig';
import LayerVisualizer from './LayerVisualizer';
import { Trash2 } from 'lucide-react';

export default function SceneNode({ id, data }) {
    const { deleteElements } = useReactFlow();

    return (
        <div className="bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700 shadow-xl min-w-[300px] relative">
            <div className="bg-slate-50 dark:bg-slate-900 px-4 py-2 border-b border-slate-200 dark:border-slate-800 rounded-t-lg font-semibold text-blue-600 dark:text-blue-400 flex items-center justify-between">
                <span>Scene Source</span>
                <div className="flex gap-2">
                    <LayerVisualizer type="scene" config={{ stars: data.stars, planets: data.planets, zodiacal: data.zodiacal }} />
                    <button
                        onClick={() => deleteElements({ nodes: [{ id }] })}
                        className="p-1 rounded hover:bg-slate-200 dark:hover:bg-slate-700 text-slate-500 dark:text-slate-400 hover:text-red-500 dark:hover:text-red-400 transition-colors"
                        title="Delete Node"
                    >
                        <Trash2 className="w-4 h-4" />
                    </button>
                </div>
            </div>
            <Handle type="source" position={Position.Right} className="!bg-blue-500 !-right-3 !w-3 !h-3" />
            <div className="p-4 max-h-[400px] overflow-y-auto custom-scrollbar nodrag nowheel">
                {/* 'nodrag' class prevents dragging the node when interacting with inputs */}
                <SceneConfig
                    stars={data.stars} setStars={data.setStars}
                    planets={data.planets} setPlanets={data.setPlanets}
                    zodiacal={data.zodiacal} setZodiacal={data.setZodiacal}
                />
            </div>
        </div>
    );
}
