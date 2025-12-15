import React, { useMemo, useState } from 'react';
import { Layers, Eye, Trash2 } from 'lucide-react';
import { useTranslation } from '../utils/i18n';
import ComponentConfigModal from './ComponentConfigModal';
import { getElementIcon } from '../utils/iconMap';

export default function LayeredView({
    nodes,
    setNodes,
    edges,
    setEdges,
    isDark,
    onInspect
}) {
    const { t } = useTranslation();
    const [selectedNode, setSelectedNode] = useState(null);
    const [draggedNode, setDraggedNode] = useState(null);
    const [isTrashActive, setIsTrashActive] = useState(false);

    // Replicate getPipeline traversal logic to order nodes strictly
    const orderedNodes = useMemo(() => {
        const roots = nodes.filter(n => n.data.type === 'scene');
        if (roots.length === 0 && nodes.length > 0) return nodes;
        if (nodes.length === 0) return [];

        let visited = new Set();
        let sorted = [];
        let queue = [...roots];

        while (queue.length > 0) {
            const node = queue.shift();
            if (visited.has(node.id)) continue;
            visited.add(node.id);
            sorted.push(node);

            const outEdges = edges.filter(e => e.source === node.id);
            const children = outEdges
                .map(e => nodes.find(n => n.id === e.target))
                .filter(n => n && !visited.has(n.id))
                .sort((a, b) => a.position.y - b.position.y);

            queue.push(...children);
        }

        nodes.forEach(n => {
            if (!visited.has(n.id)) sorted.push(n);
        });

        return sorted;
    }, [nodes, edges]);

    const handleNodeChange = (nodeId, newConfig) => {
        setNodes(nds => nds.map(n => {
            if (n.id === nodeId) {
                // Legacy Node Support: Update inner element config
                if (n.data.elements && n.data.elements.length > 0) {
                    const updatedElements = [...n.data.elements];
                    updatedElements[0] = { ...updatedElements[0], config: newConfig };
                    return { ...n, data: { ...n.data, elements: updatedElements } };
                }
                // New Component Node: Update direct config
                return { ...n, data: { ...n.data, config: newConfig } };
            }
            return n;
        }));
    };

    // Drag and Drop Handlers
    const handleDragStart = (e, node) => {
        setDraggedNode(node);
        e.dataTransfer.effectAllowed = 'move';
    };

    const handleDragOver = (e) => {
        e.preventDefault();
        e.dataTransfer.dropEffect = 'move';
    };

    const handleDropOnNode = (e, targetNode) => {
        e.preventDefault();
        if (!draggedNode || !targetNode || draggedNode.id === targetNode.id) return;

        console.log(`Swapping Node ${draggedNode.id} with ${targetNode.id}`);
        performSwap(draggedNode, targetNode);
        setDraggedNode(null);
    };

    // We need to expose `setEdges` to swapNodes. It is in props.
    // Let's redefine swapNodes to use `setEdges` correctly.
    const performSwap = (nodeA, nodeB) => {
        // Swap Edges
        const inA = edges.find(e => e.target === nodeA.id);
        const outA = edges.find(e => e.source === nodeA.id);
        const inB = edges.find(e => e.target === nodeB.id);
        const outB = edges.find(e => e.source === nodeB.id);

        let newEdges = [...edges];
        const update = (edgeId, updates) => {
            newEdges = newEdges.map(e => e.id === edgeId ? { ...e, ...updates } : e);
        };

        const isAtoB = outA?.target === nodeB.id;
        const isBtoA = outB?.target === nodeA.id;

        if (isAtoB) {
            if (inA) update(inA.id, { target: nodeB.id });
            if (outA) update(outA.id, { source: nodeB.id, target: nodeA.id });
            if (outB) update(outB.id, { source: nodeA.id });
        } else if (isBtoA) {
            if (inB) update(inB.id, { target: nodeA.id });
            if (outB) update(outB.id, { source: nodeA.id, target: nodeB.id });
            if (outA) update(outA.id, { source: nodeB.id });
        } else {
            if (inA) update(inA.id, { target: nodeB.id });
            if (outA) update(outA.id, { source: nodeB.id });
            if (inB) update(inB.id, { target: nodeA.id });
            if (outB) update(outB.id, { source: nodeA.id });
        }

        // Swap Positions
        const posA = { ...nodeA.position };
        const posB = { ...nodeB.position };
        const newNodes = nodes.map(n => {
            if (n.id === nodeA.id) return { ...n, position: posB };
            if (n.id === nodeB.id) return { ...n, position: posA };
            return n;
        });

        // Batch updates
        setNodes(newNodes);
        setEdges(newEdges);
    };

    const handleDeletePayload = (nodeToDelete) => {
        // Remove Node
        setNodes(nds => nds.filter(n => n.id !== nodeToDelete.id));

        // Heal Edges (Pre -> Deleted -> Post  ==>  Pre -> Post)
        const inEdge = edges.find(e => e.target === nodeToDelete.id);
        const outEdge = edges.find(e => e.source === nodeToDelete.id);

        let newEdges = edges.filter(e => e.source !== nodeToDelete.id && e.target !== nodeToDelete.id);

        if (inEdge && outEdge) {
            // Connect Pre to Post
            // Ensure unique ID
            const newId = `e${inEdge.source}-${outEdge.target}`;
            if (!newEdges.find(e => e.id === newId)) {
                newEdges.push({
                    id: newId,
                    source: inEdge.source,
                    target: outEdge.target,
                    animated: true,
                    type: 'default'
                });
            }
        }

        setEdges(newEdges);
    };

    return (
        <div className="flex flex-row items-stretch overflow-x-auto h-full bg-slate-100 dark:bg-slate-950/50 relative">
            {/* Trash Zone */}
            {draggedNode && (
                <div
                    className={`absolute bottom-4 right-4 z-50 p-4 rounded-full transition-all duration-300 border-2 
                        ${isTrashActive ? 'bg-red-100 border-red-500 scale-110' : 'bg-slate-200 border-slate-300 opacity-60'}`}
                    onDragOver={(e) => { e.preventDefault(); setIsTrashActive(true); }}
                    onDragLeave={() => setIsTrashActive(false)}
                    onDrop={(e) => {
                        e.preventDefault();
                        if (draggedNode) {
                            handleDeletePayload(draggedNode); // Need to define logic properly below
                            setDraggedNode(null);
                            setIsTrashActive(false);
                        }
                    }}
                >
                    <Trash2 className={`w-8 h-8 ${isTrashActive ? 'text-red-600' : 'text-slate-500'}`} />
                </div>
            )}
            {/* Component Stream with Full Height Columns */}
            <div className="flex flex-row items-stretch space-x-0 min-w-fit h-full">
                {orderedNodes.length === 0 && (
                    <div className="flex flex-col items-center justify-center w-full min-w-[300px] text-slate-400 opacity-50">
                        <Layers className="w-12 h-12 mb-2" />
                        <span className="text-sm">No components</span>
                    </div>
                )}

                {orderedNodes.map((node, index) => {
                    // Robust Data Unpacking for Legacy Compatibility
                    let effectiveData = node.data;
                    if (node.data.elements && node.data.elements.length > 0) {
                        effectiveData = { ...node.data.elements[0] };
                    }

                    const { label, type, icon } = effectiveData;
                    const iconPath = effectiveData.iconPath || getElementIcon(type);
                    const Icon = icon;

                    return (
                        <div
                            key={node.id}
                            className={`relative flex flex-col items-center min-w-[180px] w-[180px] border-r border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 group transition-opacity duration-200
                                ${draggedNode?.id === node.id ? 'opacity-50' : 'opacity-100'}
                            `}
                            draggable
                            onDragStart={(e) => handleDragStart(e, node)}
                            onDragOver={handleDragOver}
                            onDrop={(e) => {
                                e.preventDefault();
                                // Call logic to swap
                                if (draggedNode && draggedNode.id !== node.id) {
                                    performSwap(draggedNode, node);
                                    setDraggedNode(null);
                                }
                            }}
                        >
                            {/* Step Header */}
                            <div className="w-full text-center py-2 border-b border-slate-100 dark:border-slate-800">
                                <span className="text-[10px] font-bold uppercase tracking-wider text-slate-400">
                                    Step {index + 1}
                                </span>
                            </div>

                            {/* Column Content - Centered Icon */}
                            <div className="flex-1 flex flex-col items-center justify-center p-4">
                                <button
                                    onClick={() => setSelectedNode(node)}
                                    className="w-20 h-20 rounded-2xl bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 shadow-sm hover:shadow-lg hover:scale-105 hover:border-blue-500 transition-all flex flex-col items-center justify-center gap-2 group-hover:bg-white dark:group-hover:bg-slate-800"
                                    title={`Edit ${label}`}
                                >
                                    {iconPath ? (
                                        <img src={iconPath} alt={label} className="w-8 h-8 dark:invert opacity-80" />
                                    ) : Icon ? (
                                        <Icon className="w-8 h-8 text-slate-600 dark:text-slate-300" />
                                    ) : (
                                        <Layers className="w-8 h-8 text-slate-400" />
                                    )}
                                    <span className="text-[10px] font-medium text-slate-500 group-hover:text-blue-600 dark:group-hover:text-blue-400 max-w-[90%] truncate">
                                        {label}
                                    </span>
                                </button>
                            </div>

                            {/* Connection Line Indicator (Visual only) */}
                            {index < orderedNodes.length - 1 && (
                                <div className="absolute top-1/2 -right-3 w-6 h-[2px] bg-slate-200 dark:bg-slate-800 z-0 hidden" />
                            )}
                        </div>
                    );
                })}
            </div>

            {/* Modal */}
            <ComponentConfigModal
                node={selectedNode}
                isOpen={!!selectedNode}
                onClose={() => setSelectedNode(null)}
                onChange={handleNodeChange}
                onInspect={onInspect}
            />
        </div>
    );
}
