import React from 'react';
import { Handle, Position, useReactFlow } from 'reactflow';
import { Layers, Trash2 } from 'lucide-react';
import ElementRow from './ElementRow';
import SignalVisualizer from './SignalVisualizer';

export default function LayerNode({ id, data, selected }) {
    const { deleteElements, setNodes } = useReactFlow();

    // Data.elements is the list of items [ { type, config, label, icon... }, ... ]
    const elements = data.elements || [];

    const handleElementChange = (index, newElement) => {
        const newElements = [...elements];
        newElements[index] = newElement;

        setNodes((nds) => nds.map((node) => {
            if (node.id === id) {
                return { ...node, data: { ...node.data, elements: newElements } };
            }
            return node;
        }));
    };

    const handleRemoveElement = (index) => {
        const newElements = elements.filter((_, i) => i !== index);
        setNodes((nds) => nds.map((node) => {
            if (node.id === id) {
                return { ...node, data: { ...node.data, elements: newElements } };
            }
            return node;
        }));
    };

    // Calculate Capacity based on first element? Or Sum?
    // User logic: "The layer has elements". The signal enters the layer.
    // Usually the first element determines the "Input Capacity" (e.g. Fiber Injection).
    // Let's take the first element's capacity heuristic.
    const firstElem = elements[0];
    let capacity = 1;
    if (firstElem) {
        if (firstElem.config && firstElem.config.modes) capacity = firstElem.config.modes;
        // else default 1
    }

    return (
        <div className={`bg-white dark:bg-slate-800 rounded-lg border shadow-xl min-w-[320px] relative transition-colors duration-200 ${selected ? 'border-blue-500 ring-2 ring-blue-500 ring-opacity-50' : 'border-slate-200 dark:border-slate-700'}`}>
            <Handle type="target" position={Position.Left} className="!bg-blue-500 !-left-4 !w-4 !h-4" />

            {/* Layer Header */}
            <div className="bg-slate-100 dark:bg-slate-950/50 px-4 py-3 border-b border-slate-200 dark:border-slate-800 rounded-t-lg flex items-center justify-between">
                <div className="flex items-center gap-2">
                    <div className="p-1.5 bg-indigo-100 dark:bg-indigo-900/30 rounded-md text-indigo-600 dark:text-indigo-400">
                        <Layers className="w-4 h-4" />
                    </div>
                    <div>
                        <h3 className="font-semibold text-slate-800 dark:text-slate-200 text-sm">Optical Layer</h3>
                        <p className="text-[10px] text-slate-500 uppercase tracking-wider font-bold">{elements.length} Elements</p>
                    </div>
                </div>
                <button
                    onClick={() => deleteElements({ nodes: [{ id }] })}
                    className="p-1.5 rounded hover:bg-slate-200 dark:hover:bg-slate-800 text-slate-400 hover:text-red-500 transition-colors"
                    title="Delete Layer"
                >
                    <Trash2 className="w-4 h-4" />
                </button>
            </div>

            {/* Signal Visualizer (Input into the Layer) */}
            <div className="px-4 pt-4 pb-2">
                <SignalVisualizer capacity={capacity} />
            </div>

            {/* Elements List */}
            <div className="p-4 pt-0 space-y-2 max-h-[600px] overflow-y-auto custom-scrollbar nodrag nowheel">
                {elements.length === 0 && (
                    <div className="text-center p-4 border border-dashed border-slate-300 dark:border-slate-700 rounded-lg text-slate-400 text-xs italic">
                        Drop items here to add elements...
                    </div>
                )}
                {elements.map((el, idx) => (
                    <ElementRow
                        key={idx}
                        index={idx}
                        element={el}
                        onChange={handleElementChange}
                        onRemove={handleRemoveElement}
                    />
                ))}
            </div>

            <Handle type="source" position={Position.Right} className="!bg-purple-500 !-right-4 !w-4 !h-4" />
        </div>
    );
}
