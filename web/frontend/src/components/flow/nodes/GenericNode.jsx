import React from 'react';
import { Handle, Position, useReactFlow } from 'reactflow';
import NodeContent from './NodeContent';

export default function GenericNode({ id, data, selected }) {
    const { deleteElements } = useReactFlow();

    return (
        <div className="relative">
            {data.hasInput !== false && <Handle type="target" position={Position.Left} className="!bg-blue-500 !-left-4 !w-4 !h-4" />}

            <NodeContent
                id={id}
                label={data.label}
                iconPath={data.iconPath}
                Icon={data.icon}
                config={data.config}
                setConfig={data.setConfig}
                fields={data.fields}
                hasInput={data.hasInput}
                hasOutput={data.hasOutput}
                onInspect={data.onInspect}
                onDelete={(id) => deleteElements({ nodes: [{ id }] })}
                selected={selected}
                isDark={data.isDark}
            />

            {/* Ports */}
            {/* Input Port (Left) - Generic */}
            {data.hasInput !== false && (
                <Handle
                    type="target"
                    position={Position.Left}
                    id="in"
                    className={`!w-3 !h-3 !-left-3 ${data.handleClass || '!bg-slate-400'}`}
                />
            )}

            {/* Output Port (Right) - Generic */}
            {data.hasOutput !== false && (
                <Handle
                    type="source"
                    position={Position.Right}
                    id="out"
                    className={`!w-3 !h-3 !-right-3 ${data.handleClass || '!bg-slate-400'}`}
                />
            )}
        </div>
    );
}
