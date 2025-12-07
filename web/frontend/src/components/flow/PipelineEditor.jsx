import React, { useState, useCallback, useRef, useMemo } from 'react';
import ReactFlow, {
    ReactFlowProvider,
    addEdge,
    useNodesState,
    useEdgesState,
    Controls,
    Background,
    MiniMap
} from 'reactflow';
import 'reactflow/dist/style.css';

import SceneNode from './nodes/SceneNode';
import AtmosphereNode from './nodes/AtmosphereNode';
import TelescopeNode from './nodes/TelescopeNode';
import CameraNode from './nodes/CameraNode';

const nodeTypes = {
    scene: SceneNode,
    atmosphere: AtmosphereNode,
    telescope: TelescopeNode,
    camera: CameraNode
};

let id = 1;
const getId = () => `node_${id++}`;

export default function PipelineEditor({
    stars, setStars,
    planets, setPlanets,
    zodiacal, setZodiacal,
    atmosphere, setAtmosphere,
    telescope, setTelescope,
    camera, setCamera,
    runSimulation
}) {
    const reactFlowWrapper = useRef(null);
    const [reactFlowInstance, setReactFlowInstance] = useState(null);

    // Initial Nodes
    const initialNodes = [
        {
            id: 'scene-1',
            type: 'scene',
            position: { x: 50, y: 100 },
            data: { stars, setStars, planets, setPlanets, zodiacal, setZodiacal }
        },
        {
            id: 'tel-1',
            type: 'telescope',
            position: { x: 500, y: 100 },
            data: { config: telescope, setConfig: setTelescope }
        },
        {
            id: 'cam-1',
            type: 'camera',
            position: { x: 950, y: 100 },
            data: { config: camera, setConfig: setCamera }
        }
    ];

    const initialEdges = [
        { id: 'e1-2', source: 'scene-1', target: 'tel-1', animated: true },
        { id: 'e2-3', source: 'tel-1', target: 'cam-1', animated: true },
    ];

    const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
    const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);

    // Update node data when props change
    useMemo(() => {
        setNodes((nds) =>
            nds.map((node) => {
                if (node.type === 'scene') {
                    // Update data refs directly to avoid full re-render on every keystroke if possible,
                    // but ReactFlow shallow compares. We must pass fresh objects if we want updates inside nodes.
                    // This creates a render loop if not careful.
                    // Ideally, nodes should use internal state or context, but here we pass props.
                    // We only strictly need to update if the reference changes from parent.
                    node.data = { stars, setStars, planets, setPlanets, zodiacal, setZodiacal };
                } else if (node.type === 'atmosphere') {
                    node.data = { config: atmosphere, setConfig: setAtmosphere };
                } else if (node.type === 'telescope') {
                    node.data = { config: telescope, setConfig: setTelescope };
                } else if (node.type === 'camera') {
                    node.data = { config: camera, setConfig: setCamera };
                }
                return node;
            })
        );
    }, [stars, planets, zodiacal, atmosphere, telescope, camera, setNodes]);


    const onConnect = useCallback((params) => setEdges((eds) => addEdge({ ...params, animated: true }, eds)), [setEdges]);

    const onDragOver = useCallback((event) => {
        event.preventDefault();
        event.dataTransfer.dropEffect = 'move';
    }, []);

    const onDrop = useCallback(
        (event) => {
            event.preventDefault();

            const type = event.dataTransfer.getData('application/reactflow');
            if (typeof type === 'undefined' || !type) {
                return;
            }

            const position = reactFlowInstance.project({
                x: event.clientX - reactFlowWrapper.current.getBoundingClientRect().left,
                y: event.clientY - reactFlowWrapper.current.getBoundingClientRect().top,
            });

            // Populate data based on type
            let data = {};
            if (type === 'scene') {
                data = { stars, setStars, planets, setPlanets, zodiacal, setZodiacal };
            } else if (type === 'atmosphere') {
                data = { config: atmosphere, setConfig: setAtmosphere };
            } else if (type === 'telescope') {
                data = { config: telescope, setConfig: setTelescope };
            } else if (type === 'camera') {
                data = { config: camera, setConfig: setCamera };
            }

            const newNode = {
                id: getId(),
                type,
                position,
                data: data
            };

            setNodes((nds) => nds.concat(newNode));
        },
        [reactFlowInstance, stars, planets, zodiacal, atmosphere, telescope, camera, setStars, setPlanets, setZodiacal, setAtmosphere, setTelescope, setCamera, setNodes]
    );

    // Serialization for backend
    // We need to walk the graph from Scene to Camera
    const getPipeline = () => {
        // Simple BFS or finding the 'scene' node and traversing edges
        // Assuming linear chain for now

        let path = [];
        // Find Scene Node
        const sceneNode = nodes.find(n => n.type === 'scene');
        if (!sceneNode) return [];

        let current = sceneNode;
        path.push({ type: 'scene', config: { stars, planets, zodiacal } });

        // Basic traversal (handles linear only correctly)
        // A better way is using react-flow getOutgoers
        // Loop limit to prevent infinite
        for (let i = 0; i < 10; i++) {
            const outEdges = edges.filter(e => e.source === current.id);
            if (outEdges.length === 0) break;

            // Prefer first edge
            const nextNodeId = outEdges[0].target;
            const nextNode = nodes.find(n => n.id === nextNodeId);
            if (!nextNode) break;

            if (nextNode.type === 'atmosphere') {
                path.push({ type: 'atmosphere', config: atmosphere });
            } else if (nextNode.type === 'telescope') {
                path.push({ type: 'telescope', config: telescope });
            } else if (nextNode.type === 'camera') {
                path.push({ type: 'camera', config: camera });
            }

            current = nextNode;
        }
        return path;
    };

    const handleRun = () => {
        const pipeline = getPipeline();
        runSimulation(pipeline);
    };

    return (
        <div className="flex h-full flex-col">
            <div className="h-12 bg-slate-800 border-b border-slate-700 flex items-center px-4 justify-between">
                <span className="text-slate-400 text-sm">Drag nodes from the left sidebar to add them. Connect visual path.</span>
                <button onClick={handleRun} className="bg-blue-600 hover:bg-blue-500 px-4 py-1.5 rounded text-white text-sm font-bold shadow">
                    Run Pipeline
                </button>
            </div>
            <div className="flex-1" ref={reactFlowWrapper}>
                <ReactFlow
                    nodes={nodes}
                    edges={edges}
                    onNodesChange={onNodesChange}
                    onEdgesChange={onEdgesChange}
                    onConnect={onConnect}
                    onInit={setReactFlowInstance}
                    onDrop={onDrop}
                    onDragOver={onDragOver}
                    nodeTypes={nodeTypes}
                    fitView
                >
                    <Controls />
                    <Background color="#1e293b" gap={16} />
                </ReactFlow>
            </div>
        </div>
    );
}
