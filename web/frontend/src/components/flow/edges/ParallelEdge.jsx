import React from 'react';
import { getBezierPath } from 'reactflow';

export default function ParallelEdge({
    id,
    sourceX,
    sourceY,
    targetX,
    targetY,
    sourcePosition,
    targetPosition,
    style = {},
    markerEnd,
    data
}) {
    const pathCount = data?.pathCount || 1;
    const paths = [];

    // Bundle settings
    const spacing = 6; // pixels between lines

    for (let i = 0; i < pathCount; i++) {
        // Calculate offset to center the bundle
        const offset = (i - (pathCount - 1) / 2) * spacing;

        // We offset verticaly for Left/Right handles (most common)
        // If handles are Top/Bottom, we'd offset horizontally. 
        // Assuming Left/Right for now as per our Node designs.

        const [edgePath] = getBezierPath({
            sourceX,
            sourceY: sourceY + offset,
            sourcePosition,
            targetX,
            targetY: targetY + offset,
            targetPosition,
        });

        paths.push(
            <path
                key={i}
                id={`${id}_${i}`}
                style={{ ...style, strokeWidth: 1.5 }}
                className="react-flow__edge-path stroke-slate-400 dark:stroke-slate-500 hover:stroke-blue-500 transition-colors"
                d={edgePath}
                markerEnd={markerEnd}
            />
        );
    }

    return (
        <>
            {paths}
            {/* Invisible wider path for easier interaction/selection */}
            <path
                d={getBezierPath({ sourceX, sourceY, sourcePosition, targetX, targetY, targetPosition })[0]}
                style={{ strokeWidth: Math.max(20, pathCount * spacing), stroke: 'transparent', fill: 'none', cursor: 'pointer' }}
                className="react-flow__edge-interaction"
            />
        </>
    );
}
