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

    // Geometry Data from Logic
    // If not present (e.g. initial load), default to centered bundle
    const sourceTotal = data?.sourceCapacity || pathCount;
    const targetTotal = data?.targetCapacity || pathCount;

    const paths = [];

    // Bundle settings
    const spacing = 15;

    for (let i = 0; i < pathCount; i++) {
        // Calculate Active Indices (Centered Logic)
        // We select the "Middle" ports of the Source and Target to connect
        // Source Index: Starting from (Total - PathCount) / 2
        // Target Index: Starting from (Total - PathCount) / 2

        // We use Math.floor to bias towards Top/Left in odd mismatches, consistent with 0.5 logic usually
        const sourceIndex = i + Math.floor((sourceTotal - pathCount) / 2);
        const targetIndex = i + Math.floor((targetTotal - pathCount) / 2);

        // Calculate Offsets
        const sourceOffset = (sourceIndex - (sourceTotal - 1) / 2) * spacing;
        const targetOffset = (targetIndex - (targetTotal - 1) / 2) * spacing;

        const [edgePath] = getBezierPath({
            sourceX,
            sourceY: sourceY + sourceOffset,
            sourcePosition,
            targetX,
            targetY: targetY + targetOffset, // Use calculated target Y
            targetPosition,
        });

        paths.push(
            <path
                key={i}
                id={`${id}_${i}`}
                style={{
                    ...style,
                    strokeWidth: 1.5,
                    // Force consistent animation timing for all paths
                    animationDelay: '0s',
                    animationDuration: '1s'
                }}
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
