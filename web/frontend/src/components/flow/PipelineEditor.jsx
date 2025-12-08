import React, { useState, useCallback, useRef, useMemo } from 'react';
import ReactFlow, {
    ReactFlowProvider,
    addEdge,
    updateEdge,
    useNodesState,
    useEdgesState,
    Controls,
    Background,
    MiniMap
} from 'reactflow';
import 'reactflow/dist/style.css';
import { Menu, Sun, Moon, Heart, Github, Book } from 'lucide-react';

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
    runSimulation,
    onToggleSidebar,
    onToggleTheme,
    isDark
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


    // Enforce 1-to-1 connections per handle and prevent self-loops
    const isValidConnection = useCallback((connection) => {
        // Prevent self-loops
        return connection.source !== connection.target;
    }, []);

    const onConnect = useCallback((params) => {
        if (!reactFlowInstance) return;

        // Find source node to check if it supports multiple outputs (only Telescope does)
        const sourceNode = reactFlowInstance.getNode(params.source);
        const isMultiOutput = sourceNode?.type === 'telescope';

        setEdges((eds) => {
            // Enforce 1-to-1 strict rules
            // 1. Every Target input can have only ONE connection (universally true for our nodes)
            // 2. Non-Telescope Sources can have only ONE connection (Scene, Atmosphere)
            // 3. Telescope Sources can have one connection per HANDLE

            const filtered = eds.filter(e => {
                // Check Target Collision
                if (e.target === params.target) {
                    return false; // Remove existing input link to this node, regardless of handle
                }

                // Check Source Collision
                if (e.source === params.source) {
                    if (isMultiOutput) {
                        // For Telescope: Collision only if handles match
                        // (Use loose match for robustness against null/undefined)
                        const h1 = e.sourceHandle || null;
                        const h2 = params.sourceHandle || null;
                        if (h1 === h2) return false;
                    } else {
                        // For others (Scene, etc.): Collision if source node matches (Single Output)
                        return false;
                    }
                }

                return true;
            });
            return addEdge({ ...params, animated: true }, filtered);
        });
    }, [setEdges, reactFlowInstance]);

    const edgeUpdateSuccessful = useRef(true);

    const onEdgeUpdateStart = useCallback(() => {
        edgeUpdateSuccessful.current = false;
    }, []);

    const onEdgeUpdate = useCallback((oldEdge, newConnection) => {
        edgeUpdateSuccessful.current = true;
        setEdges((els) => updateEdge(oldEdge, newConnection, els));
    }, [setEdges]);
    const onEdgeUpdateEnd = useCallback((_, edge) => {
        // If the update was not successful (e.g. dropped on background), 
        // we DO NOT delete the edge anymore. It just snaps back.
        edgeUpdateSuccessful.current = true;
    }, []);

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

    const getPipeline = () => {
        let path = [];
        const sceneNode = nodes.find(n => n.type === 'scene');
        if (!sceneNode) return [];

        let current = sceneNode;
        path.push({ type: 'scene', config: { stars, planets, zodiacal } });

        for (let i = 0; i < 10; i++) {
            const outEdges = edges.filter(e => e.source === current.id);
            if (outEdges.length === 0) break;

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
            {/* TOP BAR */}
            <div className={`h-14 border-b flex items-center px-4 justify-between transition-colors ${isDark ? 'bg-slate-900 border-slate-700' : 'bg-white border-slate-200'}`}>
                <div className="flex items-center gap-4">
                    <button
                        onClick={onToggleSidebar}
                        className={`p-2 rounded-md hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors ${isDark ? 'text-slate-300' : 'text-slate-600'}`}
                        title="Toggle Sidebar"
                    >
                        <Menu className="w-5 h-5" />
                    </button>
                    <h1 className="text-xl font-bold bg-gradient-to-r from-blue-500 to-indigo-500 bg-clip-text text-transparent">
                        HELIOS <span className="text-xs font-normal text-slate-500 inline-block ml-2 align-middle">Visual Architect</span>
                    </h1>
                </div>

                <div className="flex items-center gap-2">
                    {/* Support Button */}
                    <a
                        href="https://paypal.me/vincentforiel"
                        target="_blank"
                        rel="noopener noreferrer"
                        className="p-2 rounded-full hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors text-pink-500 hover:text-pink-600"
                        title="Support the Project"
                    >
                        <Heart className="w-5 h-5 fill-current" />
                    </a>

                    {/* GitHub Button */}
                    <a
                        href="https://github.com/vforiel/helios"
                        target="_blank"
                        rel="noopener noreferrer"
                        className="p-2 rounded-full hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors text-slate-600 dark:text-slate-300 hover:text-black dark:hover:text-white"
                        title="GitHub Repository"
                    >
                        <Github className="w-5 h-5" />
                    </a>

                    {/* Documentation Button */}
                    <a
                        href="http://helios-project.rtfd.io/"
                        target="_blank"
                        rel="noopener noreferrer"
                        className="p-2 rounded-full hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors text-slate-600 dark:text-slate-300 hover:text-blue-600 dark:hover:text-blue-400"
                        title="Project Documentation"
                    >
                        <Book className="w-5 h-5" />
                    </a>

                    <div className="w-px h-6 bg-slate-200 dark:bg-slate-700 mx-1"></div>
                    <button
                        onClick={onToggleTheme}
                        className={`p-2 rounded-full hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors ${isDark ? 'text-yellow-400' : 'text-slate-600'}`}
                        title="Toggle Theme"
                    >
                        {isDark ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
                    </button>

                    <button
                        onClick={handleRun}
                        className="bg-blue-600 hover:bg-blue-500 text-white px-5 py-2 rounded-lg text-sm font-semibold shadow-md transition-transform active:scale-95 flex items-center"
                    >
                        Run Pipeline
                    </button>
                </div>
            </div>

            <div className="flex-1" ref={reactFlowWrapper}>
                <ReactFlow
                    nodes={nodes}
                    edges={edges}
                    onNodesChange={onNodesChange}
                    onEdgesChange={onEdgesChange}
                    onConnect={onConnect}
                    onEdgeUpdate={onEdgeUpdate}
                    onEdgeUpdateStart={onEdgeUpdateStart}
                    onEdgeUpdateEnd={onEdgeUpdateEnd}
                    deleteKeyCode={['Backspace', 'Delete']}
                    isValidConnection={isValidConnection}
                    onInit={setReactFlowInstance}
                    onDrop={onDrop}
                    onDragOver={onDragOver}
                    nodeTypes={nodeTypes}
                    connectionLineType="smoothstep"
                    fitView
                >
                    <Controls />
                    <Background color={isDark ? "#1e293b" : "#e2e8f0"} gap={16} />
                </ReactFlow>
            </div>
        </div>
    );
}
