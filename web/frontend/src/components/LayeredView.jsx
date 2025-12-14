import React, { useMemo } from 'react';
import { Layers } from 'lucide-react';
import ElementRow from './flow/nodes/ElementRow';
import { useTranslation } from '../utils/i18n';

export default function LayeredView({
    nodes,
    setNodes,
    edges,
    setEdges,
    isDark
}) {
    const { t } = useTranslation();

    // Replicate getPipeline traversal logic to order nodes strictly
    const orderedNodes = useMemo(() => {
        const roots = nodes.filter(n =>
            n.data.elements && n.data.elements.some(el => el.type === 'scene')
        );
        const root = roots.length > 0 ? roots[0] : nodes[0];
        if (!root) return [];

        let visited = new Set();
        let sorted = [];

        const traverse = (node) => {
            if (!node || visited.has(node.id)) return;
            visited.add(node.id);
            sorted.push(node);

            const outEdges = edges.filter(e => e.source === node.id);
            const targets = outEdges
                .map(e => nodes.find(n => n.id === e.target))
                .filter(n => n)
                .sort((a, b) => a.position.y - b.position.y);

            targets.forEach(traverse);
        };

        traverse(root);
        return sorted;
    }, [nodes, edges]);

    const handleElementChange = (nodeId, elementIndex, newElement) => {
        setNodes(nds => nds.map(n => {
            if (n.id === nodeId) {
                const newElements = [...(n.data.elements || [])];
                newElements[elementIndex] = newElement;
                return { ...n, data: { ...n.data, elements: newElements } };
            }
            return n;
        }));
    };

    const handleRemoveElement = (nodeId, elementIndex) => {
        setNodes(nds => nds.map(n => {
            if (n.id === nodeId) {
                const newElements = n.data.elements.filter((_, i) => i !== elementIndex);
                if (newElements.length === 0) return null;
                return { ...n, data: { ...n.data, elements: newElements } };
            }
            return n;
        }).filter(n => n !== null));
    };

    return (
        <div className="flex flex-row items-stretch overflow-x-auto h-full bg-slate-100 dark:bg-slate-950/50">
            <div className="flex flex-row items-stretch space-x-0 min-w-fit h-full">
                {orderedNodes.length === 0 && (
                    <div className="text-center text-slate-500 mt-20 w-full">
                        <Layers className="w-12 h-12 mx-auto mb-4 opacity-50" />
                        <p>{t ? t('validation.noLayers') : "No layers found."}</p>
                    </div>
                )}

                {orderedNodes.map((node, index) => {
                    const elements = node.data.elements || [];

                    return (
                        <div key={node.id} className="relative flex flex-row items-stretch h-full">

                            {/* Node Block - Full Height Column */}
                            <div className="min-w-[320px] w-[320px] bg-white dark:bg-slate-900 border-r border-slate-200 dark:border-slate-800 p-4 h-full overflow-y-auto custom-scrollbar">
                                <div className="flex items-center justify-between mb-4 pb-2 border-b border-slate-100 dark:border-slate-800">
                                    <span className="text-xs font-bold uppercase text-slate-400 tracking-wider">
                                        Step {index + 1}
                                    </span>
                                </div>

                                <div className="space-y-3">
                                    {elements.map((el, i) => (
                                        <ElementRow
                                            key={i}
                                            index={i}
                                            element={el}
                                            onChange={(idx, newEl) => handleElementChange(node.id, idx, newEl)}
                                            onRemove={(idx) => handleRemoveElement(node.id, idx)}
                                        />
                                    ))}
                                </div>
                            </div>
                        </div>
                    );
                })}
            </div>
        </div>
    );
}
