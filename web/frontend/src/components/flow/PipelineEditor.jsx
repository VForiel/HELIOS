import React, { useState, useCallback, useRef, useMemo, useEffect } from 'react';
import ReactFlow, {
    ReactFlowProvider,
    addEdge,
    updateEdge,
    useNodesState,
    useEdgesState,
    Controls,
    ControlButton,
    Background,
    MiniMap,
    useReactFlow,
    Panel
} from 'reactflow';
import 'reactflow/dist/style.css';
import TelescopeNode from './nodes/TelescopeNode';
import CameraNode from './nodes/CameraNode';
import GenericNode from './nodes/GenericNode';
import { Menu, Sun, Moon, Heart, Github, Book, Download, Upload, Cpu, Disc, Divide, GitFork, Zap, Activity, Hand, MousePointer2, Stars, Search, Camera, CloudFog } from 'lucide-react';

import LayerNode from './nodes/LayerNode';
import ParallelEdge from './edges/ParallelEdge';

const nodeTypes = {
    layer: LayerNode
};

const edgeTypes = {
    parallel: ParallelEdge
};

let id = 1;
const getId = () => `node_${id++}`;

// History Helper
const useUndoRedo = (initialNodes, initialEdges) => {
    const [past, setPast] = useState([]);
    const [future, setFuture] = useState([]);

    const takeSnapshot = useCallback((nodes, edges) => {
        setPast(old => {
            const newPast = [...old, { nodes, edges }];
            if (newPast.length > 50) newPast.shift(); // Limit to 50
            return newPast;
        });
        setFuture([]);
    }, []);

    const canUndo = past.length > 0;
    const canRedo = future.length > 0;

    return { past, setPast, future, setFuture, takeSnapshot, canUndo, canRedo };
};

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
    const edgeUpdateSuccessful = useRef(true);
    const [reactFlowInstance, setReactFlowInstance] = useState(null);
    const [clipboard, setClipboard] = useState(null);
    const [interactionMode, setInteractionMode] = useState('nav'); // 'nav' | 'select'

    // Initial Nodes (Layers)
    const initialNodes = [
        {
            id: 'layer-1',
            type: 'layer',
            position: { x: 50, y: 100 },
            data: {
                elements: [
                    { type: 'scene', label: 'Scene', config: { stars, planets, zodiacal }, icon: Stars }
                ]
            }
        },
        {
            id: 'layer-2',
            type: 'layer',
            position: { x: 500, y: 100 },
            data: {
                elements: [
                    { type: 'telescope', label: 'Telescope', config: telescope, icon: Search }
                ]
            }
        },
        {
            id: 'layer-3',
            type: 'layer',
            position: { x: 950, y: 100 },
            data: {
                elements: [
                    { type: 'camera', label: 'Camera', config: camera, icon: Camera }
                ]
            }
        }
    ];

    const initialEdges = [
        { id: 'e1-2', source: 'layer-1', target: 'layer-2', animated: true },
        { id: 'e2-3', source: 'layer-2', target: 'layer-3', animated: true },
    ];

    const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
    const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);

    // Sync Edges with Telescope Config (Parallel Paths)
    useEffect(() => {
        const collectorCount = telescope.collectors ? telescope.collectors.length : 1;

        setEdges(eds => eds.map(e => {
            const sourceNode = nodes.find(n => n.id === e.source);
            if (sourceNode && sourceNode.type === 'telescope') {
                return {
                    ...e,
                    type: 'parallel',
                    data: { ...e.data, pathCount: collectorCount }
                };
            }
            return e;
        }));
    }, [telescope, nodes, setEdges]);

    // Undo/Redo Hook
    const { past, setPast, future, setFuture, takeSnapshot, canUndo, canRedo } = useUndoRedo(initialNodes, initialEdges);

    const undo = useCallback(() => {
        if (!canUndo) return;
        const current = { nodes, edges };
        const previous = past[past.length - 1];
        const newPast = past.slice(0, past.length - 1);

        setPast(newPast);
        setFuture([current, ...future]);
        setNodes(previous.nodes);
        setEdges(previous.edges);
    }, [nodes, edges, past, future, canUndo, setNodes, setEdges, setPast, setFuture]);

    const redo = useCallback(() => {
        if (!canRedo) return;
        const current = { nodes, edges };
        const next = future[0];
        const newFuture = future.slice(1);

        setPast([...past, current]);
        setFuture(newFuture);
        setNodes(next.nodes);
        setEdges(next.edges);
    }, [nodes, edges, past, future, canRedo, setNodes, setEdges, setPast, setFuture]);

    // Snapshot Trigger
    const registerChange = useCallback(() => {
        takeSnapshot(nodes, edges);
    }, [nodes, edges, takeSnapshot]);

    // Copy/Paste Logic
    const handleCopy = useCallback(() => {
        const selected = nodes.filter(n => n.selected);
        if (selected.length > 0) {
            setClipboard(selected);
            console.log("Copied", selected.length, "nodes");
        }
    }, [nodes]);

    const handlePaste = useCallback(() => {
        if (!clipboard || clipboard.length === 0) return;
        registerChange(); // Snapshot before paste

        const newNodes = clipboard.map(node => {
            const newId = getId();
            return {
                ...node,
                id: newId,
                position: { x: node.position.x + 50, y: node.position.y + 50 },
                selected: true,
                data: { ...node.data } // Deep copy needed if nested, spread is shallow but okay for simple config? 
                // GenericNode checks 'fields' in data
            };
        });

        // Deselect current
        setNodes(nds => nds.map(n => ({ ...n, selected: false })).concat(newNodes));
    }, [clipboard, registerChange, setNodes]);

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
        return connection.source !== connection.target;
    }, []);

    const onConnect = useCallback((params) => {
        if (!reactFlowInstance) return;
        registerChange();

        // Edge Type Logic based on Layer Content
        const sourceNode = reactFlowInstance.getNode(params.source);
        let edgeType = 'default';
        let edgeData = { pathCount: 1 };

        if (sourceNode && sourceNode.data.elements) {
            const telescopeElem = sourceNode.data.elements.find(el => el.type === 'telescope');
            if (telescopeElem) {
                edgeType = 'parallel';
                edgeData = { pathCount: telescopeElem.config.collectors ? telescopeElem.config.collectors.length : 1 };
            }
        }

        setEdges((eds) => {
            const filtered = eds.filter(e => e.target !== params.target);
            return addEdge({ ...params, animated: true, type: edgeType, data: edgeData }, filtered);
        });
    }, [setEdges, reactFlowInstance, registerChange]);

    const onEdgeUpdateStart = useCallback(() => {
        edgeUpdateSuccessful.current = false;
    }, []);

    const onEdgeUpdate = useCallback((oldEdge, newConnection) => {
        edgeUpdateSuccessful.current = true;
        setEdges((els) => updateEdge(oldEdge, newConnection, els));
    }, [setEdges]);

    const onEdgeUpdateEnd = useCallback((_, edge) => {
        if (!edgeUpdateSuccessful.current) {
            setEdges((eds) => eds.filter((e) => e.id !== edge.id));
        }
        edgeUpdateSuccessful.current = true;
    }, [setEdges]);


    const onDragOver = useCallback((event) => {
        event.preventDefault();
        event.dataTransfer.dropEffect = 'move';
    }, []);

    const onDrop = useCallback(
        (event) => {
            event.preventDefault();
            registerChange();

            const type = event.dataTransfer.getData('application/reactflow');
            if (typeof type === 'undefined' || !type) return;

            const clientX = event.clientX;
            const clientY = event.clientY;

            const position = reactFlowInstance.project({
                x: clientX - reactFlowWrapper.current.getBoundingClientRect().left,
                y: clientY - reactFlowWrapper.current.getBoundingClientRect().top,
            });

            // Check intersection with existing nodes
            const hitBox = { x: position.x - 20, y: position.y - 20, width: 40, height: 40 };
            const intersections = reactFlowInstance.getIntersectingNodes(hitBox);

            // Define new element
            // Icons already imported

            let newElement = { type, config: {}, label: type, icon: Disc };

            if (type === 'scene') newElement = { type, label: 'Scene', config: { stars, planets, zodiacal }, icon: Stars };
            else if (type === 'atmosphere') newElement = { type, label: 'Atmosphere', config: atmosphere, icon: CloudFog };
            else if (type === 'telescope') newElement = { type, label: 'Telescope', config: telescope, icon: Search };
            else if (type === 'camera') newElement = { type, label: 'Camera', config: camera, icon: Camera };
            else if (type === 'lens') newElement = { type, label: 'Lens', config: { focal_length: 1.0 }, icon: Disc, fields: [{ name: 'focal_length', type: 'number', label: 'Focal Length', step: 0.1 }] };
            else if (type === 'beam_splitter') newElement = { type, label: 'Beam Splitter', config: { split_ratio: 0.5 }, icon: Divide, fields: [{ name: 'split_ratio', type: 'number', label: 'Split Ratio', step: 0.1 }] };
            else if (type === 'coronagraph') newElement = { type, label: 'Coronagraph', config: { type: '4quadrants' }, icon: Disc, fields: [{ name: 'type', type: 'select', label: 'Mask Type', options: [{ value: '4quadrants', label: '4-Quadrants' }, { value: 'vortex', label: 'Vortex' }] }] };
            else if (type === 'fiber_in') newElement = { type, label: 'Fiber Injection', config: { modes: 1 }, icon: Zap, fields: [{ name: 'modes', type: 'number', label: 'Modes', step: 1 }] };
            else if (type === 'fiber_out') newElement = { type, label: 'Fiber Output', config: {}, icon: Zap, fields: [] };
            else if (type === 'photonic') newElement = { type, label: 'Photonic Chip', config: { type: 'y_splitter', phase: 0.0 }, icon: Cpu, fields: [{ name: 'type', type: 'select', label: 'Component Type', options: [{ value: 'y_splitter', label: 'Y-Splitter' }, { value: 'tops', label: 'Phase Shifter' }, { value: 'mmi', label: 'MMI Coupler' }, { value: 'swap', label: 'Waveguide Crossing' }] }, { name: 'phase', type: 'number', label: 'Phase (rad)', step: 0.1 }] };


            if (intersections.length > 0) {
                // Add to existing Layer
                const targetNode = intersections[0]; // Take top one
                const newElements = [...(targetNode.data.elements || []), newElement];

                setNodes((nds) => nds.map((n) => {
                    if (n.id === targetNode.id) {
                        return { ...n, data: { ...n.data, elements: newElements } };
                    }
                    return n;
                }));
            } else {
                // Create New Layer Node
                const nodeId = getId();
                const newNode = {
                    id: nodeId,
                    type: 'layer',
                    position,
                    data: {
                        elements: [newElement]

                    }
                };
                setNodes((nds) => nds.concat(newNode));
            }
        },
        [reactFlowInstance, registerChange, setNodes, stars, telescope, camera, atmosphere]
    );

    const getPipeline = () => {
        // Traversal Logic - Refactored for Layers
        // We traverse Layers. For each Layer, we just append its Elements to the chain.
        // Parallel branching logic now applies to LAYERS.

        const startNodes = nodes.filter(n =>
            n.data.elements && n.data.elements.some(el => el.type === 'scene')
        );
        const root = startNodes.length > 0 ? startNodes[0] : nodes[0]; // Fallback

        if (!root) return [];

        let visited = new Set();

        const buildChain = (node) => {
            // 1. Current Layer Elements
            // We need to return a List of Configs.
            const layerElements = node.data.elements.map(el => ({
                type: el.type,
                config: el.config,
                metadata: { position: node.position }
            }));

            // 2. Children
            const outEdges = edges.filter(e => e.source === node.id);
            const targets = outEdges
                .map(e => nodes.find(n => n.id === e.target))
                .filter(n => n && !visited.has(n.id));

            targets.forEach(t => visited.add(t.id));

            if (targets.length === 0) return [layerElements];
            else if (targets.length === 1) {
                // Linear connection to next Layer
                return [layerElements, ...buildChain(targets[0])];
            } else {
                // Parallel Layers from this Layer
                const branches = targets.map(t => {
                    const res = buildChain(t);
                    return res;
                });
                return [layerElements, branches];
            }
        };

        visited.add(root.id);
        const configList = buildChain(root);
        return configList;
    };

    const handleRun = () => {
        const pipeline = getPipeline();
        runSimulation(pipeline);
    };

    const handleExport = async () => {
        try {
            const pipeline = getPipeline();
            // Wrap in payload
            const payload = { mode: 'pipeline', layers: pipeline };
            console.log("Export Payload:", JSON.stringify(payload, null, 2));

            const response = await fetch('/api/context/export_file', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            if (!response.ok) throw new Error("Export failed");

            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = "helios_context.json";
            document.body.appendChild(a);
            a.click();

            // Delay revocation to ensure download starts
            setTimeout(() => {
                window.URL.revokeObjectURL(url);
                document.body.removeChild(a);
            }, 1000);
        } catch (e) {
            console.error(e);
            alert("Export Failed: " + e.message);
        }
    };

    // Keyboard Listeners
    useEffect(() => {
        const handleKeyDown = (e) => {
            // Check modifier (Ctrl or Cmd)
            const meta = e.ctrlKey || e.metaKey;

            if (meta && e.key === 'z') {
                e.preventDefault();
                undo();
            } else if (meta && e.key === 'y') {
                e.preventDefault();
                redo();
            } else if (meta && e.key === 'c') {
                // Let native copy work for text? Only if no input focused.
                if (document.activeElement.tagName !== 'INPUT' && document.activeElement.tagName !== 'TEXTAREA') {
                    e.preventDefault();
                    handleCopy();
                }
            } else if (meta && e.key === 'v') {
                if (document.activeElement.tagName !== 'INPUT' && document.activeElement.tagName !== 'TEXTAREA') {
                    e.preventDefault();
                    handlePaste();
                }
            } else if (meta && e.key === 's') {
                e.preventDefault();
                handleExport(); // Use existing export logic
            } else if (meta && e.key === 'a') {
                e.preventDefault();
                setNodes(nds => nds.map(n => ({ ...n, selected: true })));
            }
        };

        window.addEventListener('keydown', handleKeyDown);
        return () => window.removeEventListener('keydown', handleKeyDown);
    }, [undo, redo, handleCopy, handlePaste]); // handleExport needs to be stable or ref

    const fileInputRef = useRef(null);

    const handleImportClick = () => {
        fileInputRef.current.click();
    };

    const handleFileChange = async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        try {
            const text = await file.text();
            const jsonData = JSON.parse(text);

            const response = await fetch('/api/context/import_file', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(jsonData)
            });

            if (!response.ok) throw new Error("Import failed");

            const result = await response.json();
            const layers = result.layers || [];

            const newNodes = [];
            const newEdges = [];

            // Helper to recursively process layers
            const processLayerList = (list, parentId, autoX) => {
                const items = Array.isArray(list) ? list : [list];
                let currentX = autoX;
                let lastId = parentId;

                items.forEach((item, idx) => {
                    if (Array.isArray(item)) {
                        // Parallel Block
                        const branches = item;
                        branches.forEach(branch => {
                            processLayerList(branch, lastId, currentX);
                        });
                        // X position doesn't increment for parallel branches typically, 
                        // but logic needs review if they merge back. For now, simple recursion.
                    } else {
                        // Single Layer
                        const layer = item;
                        const id = "node_imp_" + newNodes.length;

                        let position = { x: currentX, y: 100 + (Math.random() * 50) };
                        if (layer.metadata && layer.metadata.position) {
                            position = layer.metadata.position;
                            if (position.x >= currentX) currentX = position.x + 400;
                        } else {
                            const pNode = newNodes.find(n => n.id === parentId);
                            if (parentId && pNode) {
                                position.y = pNode.position.y;
                            }
                            currentX += 450;
                        }

                        let data = { elements: [] };
                        const type = layer.type;
                        const conf = layer.config || {};

                        // Map to Elements
                        if (type === 'scene') {
                            if (conf.stars) setStars(conf.stars);
                            if (conf.planets) setPlanets(conf.planets);
                            if (conf.zodiacal) setZodiacal(conf.zodiacal);
                            data.elements.push({ type, label: 'Scene', config: { stars: conf.stars, planets: conf.planets, zodiacal: conf.zodiacal }, icon: Stars });
                        } else if (type === 'atmosphere') {
                            setAtmosphere(conf);
                            data.elements.push({ type, label: 'Atmosphere', config: conf, icon: CloudFog });
                        } else if (type === 'telescope') {
                            setTelescope(conf);
                            data.elements.push({ type, label: 'Telescope', config: conf, icon: Search });
                        } else if (type === 'camera') {
                            setCamera(conf);
                            data.elements.push({ type, label: 'Camera', config: conf, icon: Camera });
                        } else {
                            // Generics
                            let element = { type, config: conf, label: type, icon: Disc };
                            if (type === 'lens') element = { ...element, label: 'Lens', icon: Disc, fields: [{ name: 'focal_length', type: 'number', label: 'Focal Length', step: 0.1 }] };
                            // Add more specific types as needed
                            data.elements.push(element);
                        }

                        newNodes.push({ id, type: 'layer', position, data });

                        if (parentId) {
                            const edge = {
                                id: "e_" + parentId + "_" + id,
                                source: parentId,
                                target: id,
                                animated: true,
                                type: 'default',
                                data: { pathCount: 1 }
                            };

                            // Check for parallel edge based on parent being Telescope
                            const pNode = newNodes.find(n => n.id === parentId);
                            if (pNode && pNode.data.elements) {
                                const tel = pNode.data.elements.find(el => el.type === 'telescope');
                                if (tel) {
                                    edge.type = 'parallel';
                                    edge.data.pathCount = tel.config.collectors ? tel.config.collectors.length : 1;
                                }
                            }
                            newEdges.push(edge);
                        }
                        lastId = id;
                    }
                });
            };

            processLayerList(layers, null, 50);

            setNodes(newNodes);
            setEdges(newEdges);
            e.target.value = null; // Reset input

        } catch (err) {
            console.error(err);
            alert("Import Failed: " + err.message);
        }
    };

    return (
        <div className="flex h-full flex-col">
            {/* TOP BAR */}
            <div className={"h-14 border-b flex items-center px-4 justify-between transition-colors " + (isDark ? 'bg-slate-900 border-slate-700' : 'bg-white border-slate-200')}>
                <div className="flex items-center gap-4">
                    <button
                        onClick={onToggleSidebar}
                        className={"p-2 rounded-md hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors " + (isDark ? 'text-slate-300' : 'text-slate-600')}
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

                    {/* Import/Export */}
                    <input
                        type="file"
                        ref={fileInputRef}
                        style={{ display: 'none' }}
                        accept=".json"
                        onChange={handleFileChange}
                    />

                    <button
                        onClick={handleImportClick}
                        className="p-2 rounded-full hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors text-slate-600 dark:text-slate-300 hover:text-blue-600 dark:hover:text-blue-400"
                        title="Import Context"
                    >
                        <Upload className="w-5 h-5" />
                    </button>

                    <button
                        onClick={handleExport}
                        className="p-2 rounded-full hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors text-slate-600 dark:text-slate-300 hover:text-blue-600 dark:hover:text-blue-400"
                        title="Export Context"
                    >
                        <Download className="w-5 h-5" />
                    </button>

                    <div className="w-px h-6 bg-slate-200 dark:bg-slate-700 mx-1"></div>
                    <button
                        onClick={onToggleTheme}
                        className={"p-2 rounded-full hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors " + (isDark ? 'text-yellow-400' : 'text-slate-600')}
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
                    edgeTypes={edgeTypes}
                    connectionLineType="smoothstep"
                    selectionOnDrag={interactionMode === 'select'}
                    panOnDrag={interactionMode === 'nav' || [1, 2]}
                    selectionMode="partial"
                    fitView
                    minZoom={0.1}
                >
                    <Controls>
                        <ControlButton
                            onClick={() => setInteractionMode('nav')}
                            title="Navigation Mode (Pan)"
                            className={interactionMode === 'nav' ? 'text-blue-600 bg-blue-100 font-bold' : ''}
                        >
                            <Hand className="w-4 h-4 p-0.5" />
                        </ControlButton>
                        <ControlButton
                            onClick={() => setInteractionMode('select')}
                            title="Selection Mode (Box Select)"
                            className={interactionMode === 'select' ? 'text-blue-600 bg-blue-100 font-bold' : ''}
                        >
                            <MousePointer2 className="w-4 h-4 p-0.5" />
                        </ControlButton>
                    </Controls>
                    <Background color={isDark ? "#1e293b" : "#e2e8f0"} gap={16} />
                </ReactFlow>
            </div>
        </div>
    );
}
