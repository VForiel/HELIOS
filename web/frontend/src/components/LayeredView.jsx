import React, { useMemo, useState } from 'react';
import { Layers, Eye } from 'lucide-react';
import { useTranslation } from '../utils/i18n';
import ComponentConfigModal from './ComponentConfigModal';
import { getElementIcon } from '../utils/iconMap';

export default function LayeredView({
    nodes,
    setNodes,
    edges,
    isDark,
    onInspect
}) {
    const { t } = useTranslation();
    const [selectedNode, setSelectedNode] = useState(null);

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

    return (
        <div className="flex flex-row items-stretch overflow-x-auto h-full bg-slate-100 dark:bg-slate-950/50">
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
                        <div key={node.id} className="relative flex flex-col items-center min-w-[180px] w-[180px] border-r border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 group">
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
