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

    // Group nodes into connected pipelines and then by depth
    const pipelines = useMemo(() => {
        if (nodes.length === 0) return [];

        const visited = new Set();
        const pipelines = [];

        // Helper to find connected components (Undirected for grouping)
        const findComponent = (startNode) => {
            const componentNodes = [];
            const queue = [startNode];
            const localVisited = new Set();
            localVisited.add(startNode.id);
            visited.add(startNode.id);

            // BFS for connectivity (both directions)
            while (queue.length > 0) {
                const node = queue.shift();
                componentNodes.push(node);

                // Outgoing
                edges.filter(e => e.source === node.id).forEach(e => {
                    const target = nodes.find(n => n.id === e.target);
                    if (target && !localVisited.has(target.id)) {
                        localVisited.add(target.id);
                        visited.add(target.id);
                        queue.push(target);
                    }
                });

                // Incoming
                edges.filter(e => e.target === node.id).forEach(e => {
                    const source = nodes.find(n => n.id === e.source);
                    if (source && !localVisited.has(source.id)) {
                        localVisited.add(source.id);
                        visited.add(source.id);
                        queue.push(source);
                    }
                });
            }
            return componentNodes;
        };

        // Helper to sort component nodes into layers (steps)
        const organizePipeline = (componentNodes) => {
            if (componentNodes.length === 0) return [];

            // 1. Calculate In-Degree to find roots within this component
            const inDegree = new Map();
            componentNodes.forEach(n => inDegree.set(n.id, 0));

            // Only consider edges internal to this component
            const componentNodeIds = new Set(componentNodes.map(n => n.id));
            const internalEdges = edges.filter(e => componentNodeIds.has(e.source) && componentNodeIds.has(e.target));

            internalEdges.forEach(e => {
                inDegree.set(e.target, (inDegree.get(e.target) || 0) + 1);
            });

            // 2. Identify Roots (Scene or Degree 0)
            let roots = componentNodes.filter(n => (inDegree.get(n.id) === 0) || n.data.type === 'scene');

            // Cycle fallback: if no roots but nodes exist, pick one arbitrarily (e.g. top-left)
            if (roots.length === 0 && componentNodes.length > 0) {
                roots = [componentNodes.sort((a, b) => a.position.x - b.position.x)[0]];
            }

            // 3. Assign Depth (Longest Path) via BFS/Topological Sort approach
            const depth = new Map();
            roots.forEach(r => depth.set(r.id, 0));

            // We need to process in topological order. 
            // Kahn's algorithm or just relaxed updates since graph is small.
            // Relaxed updates: Iterate edges and update target depth. 
            // Max iterations = num nodes.

            let changed = true;
            let iterations = 0;
            while (changed && iterations < componentNodes.length) {
                changed = false;
                internalEdges.forEach(e => {
                    const dSource = depth.get(e.source);
                    if (dSource !== undefined) {
                        const dTarget = depth.get(e.target);
                        if (dTarget === undefined || dTarget < dSource + 1) {
                            depth.set(e.target, dSource + 1);
                            changed = true;
                        }
                    }
                });
                iterations++;
            }

            // Fill undefined depths (unreachable islands within component)
            componentNodes.forEach(n => {
                if (!depth.has(n.id)) depth.set(n.id, 0);
            });

            // 4. Group by Depth
            const maxDepth = Math.max(...Array.from(depth.values()));
            const layers = Array.from({ length: maxDepth + 1 }, () => []);

            componentNodes.forEach(n => {
                const d = depth.get(n.id);
                layers[d].push(n);
            });

            // 5. Sort within layers by Y per usual
            layers.forEach(layer => layer.sort((a, b) => a.position.y - b.position.y));

            return layers;
        };


        // Main Logic: Find Components -> Organize them
        // 1. Find explicit roots used for ordering pipelines
        // Actually, just find components from unvisited nodes.

        // Preference: Start with 'scene' nodes to ensure they are first pipelines
        const sceneNodes = nodes.filter(n => n.data.type === 'scene').sort((a, b) => a.position.y - b.position.y);

        sceneNodes.forEach(startNode => {
            if (!visited.has(startNode.id)) {
                const component = findComponent(startNode);
                pipelines.push(organizePipeline(component));
            }
        });

        // Catch leftovers
        nodes.forEach(node => {
            if (!visited.has(node.id)) {
                const component = findComponent(node);
                pipelines.push(organizePipeline(component));
            }
        });

        return pipelines;
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

    // Calculate max steps for global header grid
    // Use memo to avoid recalculation on every render
    const maxSteps = useMemo(() => {
        if (pipelines.length === 0) return 0;
        return Math.max(...pipelines.map(p => p.length));
    }, [pipelines]);

    return (
        <div className="flex flex-col h-full bg-slate-100 dark:bg-slate-950/50 overflow-auto relative custom-scrollbar">
            {/* Trash Zone */}
            {draggedNode && (
                <div
                    className={`fixed bottom-8 right-8 z-50 p-4 rounded-full transition-all duration-300 border-2 shadow-xl
                        ${isTrashActive ? 'bg-red-100 border-red-500 scale-110' : 'bg-slate-200 border-slate-300 opacity-60'}`}
                    onDragOver={(e) => { e.preventDefault(); setIsTrashActive(true); }}
                    onDragLeave={() => setIsTrashActive(false)}
                    onDrop={(e) => {
                        e.preventDefault();
                        if (draggedNode) {
                            handleDeletePayload(draggedNode);
                            setDraggedNode(null);
                            setIsTrashActive(false);
                        }
                    }}
                >
                    <Trash2 className={`w-8 h-8 ${isTrashActive ? 'text-red-600' : 'text-slate-500'}`} />
                </div>
            )}

            {pipelines.length === 0 && (
                <div className="flex flex-col items-center justify-center h-full w-full text-slate-400 opacity-50">
                    <Layers className="w-12 h-12 mb-2" />
                    <span className="text-sm">No components</span>
                </div>
            )}

            {/* Content Grid */}
            {pipelines.length > 0 && (
                <div className="min-w-fit">
                    {/* Global Sticky Header */}
                    <div className="flex flex-row sticky top-0 z-20 bg-slate-100/95 dark:bg-slate-950/95 backdrop-blur border-b border-slate-200 dark:border-slate-800 shadow-sm">
                        {/* Pipeline Name Column Header Placeholder */}
                        <div className="w-[100px] shrink-0 p-2 font-bold text-xs text-slate-400 uppercase tracking-wider flex items-center justify-center border-r border-slate-200 dark:border-slate-800">
                            Pipeline
                        </div>
                        {Array.from({ length: maxSteps }).map((_, i) => (
                            <div key={i} className="min-w-[180px] w-[180px] shrink-0 text-center py-2 border-r border-slate-200 dark:border-slate-800 last:border-r-0">
                                <span className="text-[10px] font-bold uppercase tracking-wider text-slate-500 dark:text-slate-400">
                                    Step {i + 1}
                                </span>
                            </div>
                        ))}
                    </div>

                    {/* Pipelines */}
                    {pipelines.map((pipeline, pIndex) => (
                        <div key={pIndex} className="flex flex-row bg-white/50 dark:bg-slate-900/50 border-b border-slate-200 dark:border-slate-800">
                            {/* Pipeline Label Left Column */}
                            <div className="w-[100px] shrink-0 p-2 font-bold text-[10px] text-slate-400 uppercase tracking-wider flex items-center justify-center border-r border-slate-200 dark:border-slate-800 bg-slate-50/50 dark:bg-slate-900/50">
                                #{pIndex + 1}
                            </div>

                            {/* Steps Loop (Cells) */}
                            {pipeline.map((stepNodes, index) => (
                                <div key={index} className="min-w-[180px] w-[180px] shrink-0 border-r border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 group flex flex-col p-2 gap-2">
                                    {/* Parallel Components Loop */}
                                    {stepNodes.map((node) => {
                                        // Robust Data Unpacking
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
                                                className={`relative flex flex-col items-center bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700 shadow-sm transition-opacity duration-200
                                                    ${draggedNode?.id === node.id ? 'opacity-50' : 'opacity-100'}
                                                `}
                                                draggable
                                                onDragStart={(e) => handleDragStart(e, node)}
                                                onDragOver={handleDragOver}
                                                onDrop={(e) => {
                                                    e.preventDefault();
                                                    if (draggedNode && draggedNode.id !== node.id) {
                                                        performSwap(draggedNode, node);
                                                        setDraggedNode(null);
                                                    }
                                                }}
                                            >
                                                <div className="flex-1 flex flex-col items-center justify-center p-2 w-full">
                                                    <button
                                                        onClick={() => setSelectedNode(node)}
                                                        className="w-full flex items-center gap-2 hover:text-blue-500 transition-colors"
                                                        title={`Edit ${label}`}
                                                    >
                                                        {iconPath ? (
                                                            <img src={iconPath} alt={label} className="w-5 h-5 dark:invert opacity-80" />
                                                        ) : Icon ? (
                                                            <Icon className="w-5 h-5 text-slate-600 dark:text-slate-300" />
                                                        ) : (
                                                            <Layers className="w-5 h-5 text-slate-400" />
                                                        )}
                                                        <span className="text-[10px] font-medium text-slate-600 dark:text-slate-300 truncate text-left flex-1">
                                                            {label}
                                                        </span>
                                                    </button>
                                                </div>

                                                {/* Visual Connection (Only if not last step) */}
                                                {/* Logic roughly for first node in stack? Or just generic arrow? 
                                                    Ideally lines should connect nodes, but that requires SVG overlay. 
                                                    For now, existing logic was "Right". 
                                                    If multiple, maybe just show arrow on cell right?
                                                 */}
                                            </div>
                                        );
                                    })}

                                    {/* Connector (Visual for the whole step) */}
                                    {index < pipeline.length - 1 && (
                                        <div className="absolute top-1/2 -right-3 w-6 h-[2px] bg-slate-300 dark:bg-slate-700 z-10 hidden" />
                                    )}
                                    {/* ^ Logic is messy with relative parallel nodes. Hiding connection line for simplicity as row borders imply flow. */}
                                </div>
                            ))}

                            {/* Fill remaining space if pipeline is shorter than max */}
                            {Array.from({ length: maxSteps - pipeline.length }).map((_, i) => (
                                <div key={`empty-${i}`} className="min-w-[180px] w-[180px] shrink-0 border-r border-slate-200/50 dark:border-slate-800/50" />
                            ))}
                        </div>
                    ))}
                </div>
            )}

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
