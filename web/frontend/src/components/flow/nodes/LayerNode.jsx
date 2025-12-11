import React from 'react';
import { Handle, Position, useReactFlow } from 'reactflow';
import { Layers, Trash2 } from 'lucide-react';
import ElementRow from './ElementRow';
import SignalVisualizer from './SignalVisualizer';

export default function LayerNode({ id, data, selected }) {
    const { deleteElements, setNodes } = useReactFlow();

    // Data.elements is the list of items [ { type, config, label, icon... }, ... ]
    const elements = data.elements || [];
    const io = data.io || { incoming: 0, capacity: 0, outgoing: 0, status: 'unknown' };

    const handleElementChange = (index, newElement) => {
        const newElements = [...elements];
        newElements[index] = newElement;

        setNodes((nds) => nds.map((node) => {
            if (node.id === id) {
                // Keep IO but update elements to trigger useEffect recalculation
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

    // Input Port Props
    const inputPortCount = Math.max(io.incoming || 0, io.capacity || 0);
    const paramsInput = {
        total: inputPortCount,
        capacity: io.capacity || 0,
        incoming: io.incoming || 0
    };

    // Output Port Props
    const outputTotal = io.outputCapacity || io.outgoing || 0;
    const paramsOutput = {
        total: outputTotal,
        capacity: outputTotal,
        incoming: io.outgoing || 0
    };

    return (
        <div className={`bg-white dark:bg-slate-800 rounded-lg border shadow-xl min-w-[320px] relative transition-colors duration-200 ${selected ? 'border-blue-500 ring-2 ring-blue-500 ring-opacity-50' : 'border-slate-200 dark:border-slate-700'}`}>

            {/* --------------------------------------------------------------------------------
               ABSOLUTE POSITIONED INTERFACE LAYERS (CENTERED)
               Moved out of flow to ensure they are always vertically centered on the block.
            --------------------------------------------------------------------------------- */}

            {/* Input Interface (Left) */}
            {/* Main Clickable Handle: Moved further out (-left-7) and styled distinctly */}
            {/* We wrap Handle in a div to enforce center positioning relative to the Node Block */}
            {/* Input Interface (Left) */}
            <div className="absolute top-1/2 -left-7 -translate-y-1/2 z-50">
                <Handle
                    type="target"
                    position={Position.Left}
                    className="!static !w-auto !h-auto !min-w-[24px] !min-h-[24px] !p-1 !rounded-md !border-2 !border-slate-300 dark:!border-slate-600 !bg-slate-100 dark:!bg-slate-900 hover:!border-blue-500 hover:!bg-blue-50 dark:hover:!bg-slate-800 transition-colors flex items-center justify-center"
                    title="Input Connection"
                    style={{ transform: 'none' }}
                >
                    <SignalVisualizer config={paramsInput} layout="vertical" spacing={15} isInput={true} />
                </Handle>
            </div>

            {/* Output Interface (Right) */}
            <div className="absolute top-1/2 -right-7 -translate-y-1/2 z-50">
                <Handle
                    type="source"
                    position={Position.Right}
                    className="!static !w-auto !h-auto !min-w-[24px] !min-h-[24px] !p-1 !rounded-md !border-2 !border-slate-300 dark:!border-slate-600 !bg-slate-100 dark:!bg-slate-900 hover:!border-purple-500 hover:!bg-purple-50 dark:hover:!bg-slate-800 transition-colors flex items-center justify-center"
                    title="Output Connection"
                    style={{ transform: 'none' }}
                >
                    <SignalVisualizer config={paramsOutput} layout="vertical" spacing={15} isInput={false} />
                </Handle>
            </div>

            {/* --------------------------------------------------------------------------------
               NODE CONTENT
            --------------------------------------------------------------------------------- */}

            {/* Layer Header */}
            <div className="bg-slate-100 dark:bg-slate-950/50 px-4 py-3 border-b border-slate-200 dark:border-slate-800 rounded-t-lg flex items-center justify-between">
                <div className="flex items-center gap-2">
                    <div className="p-1.5 bg-indigo-100 dark:bg-indigo-900/30 rounded-md text-indigo-600 dark:text-indigo-400">
                        <Layers className="w-4 h-4" />
                    </div>
                    <div>
                        <h3 className="font-semibold text-slate-800 dark:text-slate-200 text-sm">Optical Layer</h3>
                        <p className="text-[10px] text-slate-500 uppercase tracking-wider font-bold">
                            {elements.length} Elements | {io.incoming || 0} In / {io.outgoing || 0} Out
                        </p>
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

            {/* Elements List */}
            <div className="p-4 space-y-2 max-h-[600px] overflow-y-auto custom-scrollbar nodrag nowheel">
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
        </div>
    );
}
