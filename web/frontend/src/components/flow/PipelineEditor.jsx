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
import SceneNode from './nodes/SceneNode';
import AtmosphereNode from './nodes/AtmosphereNode';
import TelescopeNode from './nodes/TelescopeNode';
import CameraNode from './nodes/CameraNode';
import GenericNode from './nodes/GenericNode';
import { Menu, Sun, Moon, Heart, Github, Book, Download, Upload, Cpu, Disc, Divide, GitFork, Zap, Activity, Hand, MousePointer2 } from 'lucide-react';

const nodeTypes = {
    scene: SceneNode,
    atmosphere: AtmosphereNode,
    telescope: TelescopeNode,
    camera: CameraNode,
    lens: GenericNode,
    beam_splitter: GenericNode,
    coronagraph: GenericNode,
    fiber_in: GenericNode,
    fiber_out: GenericNode,
    photonic: GenericNode
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
    const [reactFlowInstance, setReactFlowInstance] = useState(null);
    const [clipboard, setClipboard] = useState(null);
    const [interactionMode, setInteractionMode] = useState('nav'); // 'nav' | 'select'

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
        // Prevent self-loops
        return connection.source !== connection.target;
    }, []);

    const onConnect = useCallback((params) => {
        if (!reactFlowInstance) return;
        registerChange(); // Snapshot

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

                        // Strict Replacement:
                        // 1. If handles match, replace.
                        // 2. If existing edge has NO handle (legacy/bug), replace it to clean up.
                        if (h1 === h2 || !h1) return false;
                    } else {
                        // For others (Scene, etc.): Collision if source node matches (Single Output)
                        return false;
                    }
                }

                return true;
            });
            return addEdge({ ...params, animated: true }, filtered);
        });
    }, [setEdges, reactFlowInstance, registerChange]);

    const edgeUpdateSuccessful = useRef(true);

    const onEdgeUpdateStart = useCallback(() => {
        edgeUpdateSuccessful.current = false;
    }, []);

    const onEdgeUpdate = useCallback((oldEdge, newConnection) => {
        edgeUpdateSuccessful.current = true;
        registerChange(); // Snapshot
        setEdges((els) => updateEdge(oldEdge, newConnection, els));
    }, [setEdges, registerChange]);

    const onEdgeUpdateEnd = useCallback((_, edge) => {
        // If the update was not successful (e.g. dropped on background), 
        // we DO NOT delete the edge anymore. It just snaps back.
        edgeUpdateSuccessful.current = true;
    }, []);

    const onDragOver = useCallback((event) => {
        event.preventDefault();
        event.dataTransfer.dropEffect = 'move';
    }, []);

    const updateNodeConfig = useCallback((id, newConfig) => {
        setNodes((nds) => nds.map((node) => {
            if (node.id === id) {
                return { ...node, data: { ...node.data, config: newConfig } };
            }
            return node;
        }));
    }, [setNodes]);

    const onDrop = useCallback(
        (event) => {
            event.preventDefault();
            registerChange(); // Snapshot

            const type = event.dataTransfer.getData('application/reactflow');
            if (typeof type === 'undefined' || !type) {
                return;
            }

            const position = reactFlowInstance.project({
                x: event.clientX - reactFlowWrapper.current.getBoundingClientRect().left,
                y: event.clientY - reactFlowWrapper.current.getBoundingClientRect().top,
            });

            const nodeId = getId();
            let data = {};
            const setConfig = (c) => updateNodeConfig(nodeId, c);

            // Defaults
            if (type === 'scene') {
                data = { stars, setStars, planets, setPlanets, zodiacal, setZodiacal };
            } else if (type === 'atmosphere') {
                data = { config: atmosphere, setConfig: setAtmosphere };
            } else if (type === 'telescope') {
                data = { config: telescope, setConfig: setTelescope };
            } else if (type === 'camera') {
                data = { config: camera, setConfig: setCamera };
            }
            // New Types using GenericNode
            else if (type === 'lens') {
                data = {
                    label: 'Lens', icon: Disc, isDark,
                    config: { focal_length: 1.0 },
                    fields: [{ name: 'focal_length', type: 'number', label: 'Focal Length (m)', step: 0.1 }],
                    setConfig, hasInput: true, hasOutput: true
                };
            } else if (type === 'beam_splitter') {
                data = {
                    label: 'Beam Splitter', icon: Divide, isDark,
                    config: { split_ratio: 0.5 },
                    fields: [{ name: 'split_ratio', type: 'number', label: 'Split Ratio (0-1)', step: 0.1 }],
                    setConfig, hasInput: true, hasOutput: true
                };
            } else if (type === 'coronagraph') {
                data = {
                    label: 'Coronagraph', icon: Disc, isDark,
                    config: { type: '4quadrants' },
                    fields: [{ name: 'type', type: 'select', label: 'Mask Type', options: [{ value: '4quadrants', label: '4-Quadrants' }, { value: 'vortex', label: 'Vortex' }] }],
                    setConfig, hasInput: true, hasOutput: true
                };
            } else if (type === 'fiber_in') {
                data = {
                    label: 'Fiber Injection', icon: Zap, isDark,
                    config: { modes: 1 },
                    fields: [{ name: 'modes', type: 'number', label: 'Modes', step: 1 }],
                    setConfig, hasInput: true, hasOutput: true
                };
            } else if (type === 'fiber_out') {
                data = {
                    label: 'Fiber Output', icon: Zap, isDark,
                    config: {},
                    fields: [],
                    setConfig, hasInput: true, hasOutput: true
                };
            } else if (type === 'photonic') {
                data = {
                    label: 'Photonic Chip', icon: Cpu, isDark,
                    config: { type: 'y_splitter', phase: 0.0 }, // default
                    fields: [
                        {
                            name: 'type', type: 'select', label: 'Component Type', options: [
                                { value: 'y_splitter', label: 'Y-Splitter' },
                                { value: 'tops', label: 'Phase Shifter' },
                                { value: 'mmi', label: 'MMI Coupler' },
                                { value: 'swap', label: 'Waveguide Crossing' }
                            ]
                        },
                        { name: 'phase', type: 'number', label: 'Phase (rad)', step: 0.1 }
                    ],
                    setConfig, hasInput: true, hasOutput: true
                };
            }

            const newNode = {
                id: nodeId,
                type,
                position,
                data: data
            };

            setNodes((nds) => nds.concat(newNode));
        },
        [reactFlowInstance, stars, planets, zodiacal, atmosphere, telescope, camera, setStars, setPlanets, setZodiacal, setAtmosphere, setTelescope, setCamera, setNodes, updateNodeConfig, isDark, registerChange]
    );

    const getPipeline = () => {
        // Advanced Traversal for Parallel Branches (Fan-Out)
        // 1. Identify Start (Scene)
        // 2. Traverse. If multiple outgoing edges to unvisited nodes -> Parallel Block

        const sceneNode = nodes.find(n => n.type === 'scene');
        if (!sceneNode) return [];

        let configList = [];
        let visited = new Set();

        // Recursive Helper
        const traverse = (node) => {
            if (visited.has(node.id)) return null;
            visited.add(node.id);

            // Create Config
            let layerType = node.type;
            let layerConfig = {};
            if (layerType === 'scene') layerConfig = { stars, planets, zodiacal };
            else if (layerType === 'atmosphere') layerConfig = atmosphere;
            else if (layerType === 'telescope') layerConfig = telescope;
            else if (layerType === 'camera') layerConfig = camera;
            else layerConfig = node.data.config;

            const myConfig = { type: layerType, config: layerConfig, metadata: { position: node.position } };

            // Find Next Nodes
            const outEdges = edges.filter(e => e.source === node.id);
            const targets = outEdges
                .map(e => nodes.find(n => n.id === e.target))
                .filter(n => n && !visited.has(n.id));

            // Deduplicate targets (just in case multiple edges point to same node)
            const uniqueTargets = [...new Set(targets)];

            if (uniqueTargets.length === 0) {
                return myConfig;
            } else if (uniqueTargets.length === 1) {
                // Linear
                const next = traverse(uniqueTargets[0]);
                if (next) return [myConfig, ... (Array.isArray(next) ? next : [next])]; // Flatten linear chain
                return myConfig;
            } else {
                // Branching / Parallel
                // We return current node, followed by a List of parallel branches
                // BUT: PipelineRequest structure is flat list of layers/lists.
                // So: [Current, [Branch1, Branch2, ...]]
                // Wait, if we return [A, [B, C]], this structure means A is followed by parallel B and C.

                const branches = uniqueTargets.map(t => {
                    // Traverse each branch independently
                    // Note: If branches merge later, this simple tree approach duplicates or fails. 
                    // Assuming Tree structure for now (Fan-Out Only).
                    const branchResult = traverse(t);
                    // Flatten branch result if it's a list (linear branch) -> [Node1, Node2]
                    // If it's a single node -> Node
                    return Array.isArray(branchResult) ? branchResult : [branchResult];
                });

                // Ideally we want to flatten the result into the main list if possible, but here we are returning structure.
                // Problem: 'traverse' returns a chain.
                // Let's change strategy: Linear accumulation with explicit 'Parallel Block' insertion.
            }
        };

        // Iterative Strategy for simplicity and control
        let queue = [sceneNode];
        visited = new Set([sceneNode.id]);

        // This is tricky because we need to construct the Nested JSON.
        // Let's do a tailored recursion that returns the LIST structure expected by backend.

        const buildChain = (node) => {
            // 1. Build Payload for Current Node
            let layerType = node.type;
            let layerConfig = {};
            if (layerType === 'scene') layerConfig = { stars, planets, zodiacal };
            else if (layerType === 'atmosphere') layerConfig = atmosphere;
            else if (layerType === 'telescope') layerConfig = telescope;
            else if (layerType === 'camera') layerConfig = camera;
            else layerConfig = node.data.config;

            const currentItem = { type: layerType, config: layerConfig, metadata: { position: node.position } };

            // 2. Find Children
            const outEdges = edges.filter(e => e.source === node.id);
            const targets = outEdges
                .map(e => nodes.find(n => n.id === e.target))
                .filter(n => n); // don't filter visited here for Graph logic, but for Tree yes.

            // Filter targets that we haven't processed yet?
            // For simple Fan-Out Pipeline (Tree):
            const children = targets.filter(t => !visited.has(t.id));
            children.forEach(c => visited.add(c.id));

            if (children.length === 0) {
                return [currentItem];
            } else if (children.length === 1) {
                // Sequence
                return [currentItem, ...buildChain(children[0])];
            } else {
                // Parallel
                // [Current, [Branch1, Branch2]]
                const branches = children.map(c => {
                    // Find edge to get sourceHandle
                    const edge = edges.find(e => e.source === node.id && e.target === c.id);
                    const sourceHandle = edge ? edge.sourceHandle : null;

                    let res = buildChain(c);

                    // Inject sourceHandle into the child's metadata (whether it's a single config or list)
                    let targetNodeConf = null;
                    if (Array.isArray(res)) {
                        targetNodeConf = res[0];
                    } else {
                        targetNodeConf = res;
                    }

                    if (targetNodeConf && sourceHandle) {
                        if (!targetNodeConf.metadata) targetNodeConf.metadata = {};
                        targetNodeConf.metadata.sourceHandle = sourceHandle;
                    }

                    // Flatten if single element (common case: Telescope -> Cameras)
                    if (Array.isArray(res) && res.length === 1) return res[0];
                    return res;
                });
                return [currentItem, branches];
            }
        };

        visited.clear();
        visited.add(sceneNode.id);
        configList = buildChain(sceneNode);

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
            // result is PipelineRequest { mode, layers }

            // Rebuild State
            const layers = result.layers || [];

            const newNodes = [];
            const newEdges = [];
            let xPos = 50;
            let lastNodeId = null;

            // Clear current selection/highlight if any (omitted)

            // Process layers
            // Recursive Rebuild
            const processLayerList = (layerListObj, parentId, autoX) => {
                const layerList = Array.isArray(layerListObj) ? layerListObj : [layerListObj];
                let localX = autoX;
                let lastId = parentId;

                layerList.forEach((item, idx) => {
                    if (Array.isArray(item)) {
                        // Parallel Block! 
                        const branches = item;
                        let branchY = 100;
                        if (lastId) {
                            const pNode = newNodes.find(n => n.id === lastId);
                            if (pNode) branchY = pNode.position.y - ((branches.length - 1) * 100 * 0.5);
                        }

                        branches.forEach((branch, bIdx) => {
                            // Branch is a List of layers
                            // Vertical offset for branches
                            processLayerList(branch, lastId, localX);
                        });

                    } else {
                        // Single Layer
                        const layer = item;
                        const id = `node_imp_${newNodes.length}`;

                        // Position Logic
                        let position;
                        if (layer.metadata && layer.metadata.position) {
                            position = layer.metadata.position;
                            if (position.x >= localX) localX = position.x + 400;
                        } else {
                            // Auto Layout
                            position = { x: localX, y: 100 + (Math.random() * 50) };
                            const pNode = newNodes.find(n => n.id === parentId);
                            if (parentId && pNode) {
                                position.y = pNode.position.y + ((idx - (layerList.length - 1) / 2) * 200);
                                // Simple spacing if previous was branch parent? 
                                // Actually for linear chain inside branch, use parent Y.
                                // Only for the start of branch we need offset.
                                // Let's refine:
                                if (layerList.length > 1) {
                                    // We are at start of parallel list? No.
                                    // item is 'layer'. layerList is the branch array? No.
                                    // If we are in processLayerList(branch), then layerList is the branch content.
                                    // It's linear.
                                    // Wait, processLayerList iterates linear list.
                                }
                            }
                            localX += 450;
                        }

                        let type = layer.type;
                        let data = {};

                        // ... Config Mapping ...
                        if (type === 'scene') {
                            const conf = layer.config;
                            if (conf.stars) setStars(conf.stars);
                            if (conf.planets) setPlanets(conf.planets);
                            if (conf.zodiacal) setZodiacal(conf.zodiacal);
                        } else if (type === 'atmosphere') {
                            setAtmosphere(layer.config);
                        } else if (type === 'telescope') {
                            setTelescope(layer.config);
                        } else if (type === 'camera') {
                            setCamera(layer.config);
                        } else {
                            // Generics
                            const conf = layer.config;
                            const setConfig = (c) => updateNodeConfig(id, c);
                            data = { config: conf, setConfig, hasInput: true, hasOutput: true, isDark };

                            if (type === 'lens') data = { ...data, label: 'Lens', icon: Disc, fields: [{ name: 'focal_length', type: 'number', label: 'Focal Length', step: 0.1 }] };
                            else if (type === 'beam_splitter') data = { ...data, label: 'Beam Splitter', icon: Divide, fields: [{ name: 'split_ratio', type: 'number', label: 'Split Ratio', step: 0.1 }] };
                            else if (type === 'coronagraph') data = { ...data, label: 'Coronagraph', icon: Disc, fields: [{ name: 'type', type: 'select', label: 'Mask Type', options: [{ value: '4quadrants', label: '4-Quadrants' }, { value: 'vortex', label: 'Vortex' }] }] };
                            else if (type === 'fiber_in') data = { ...data, label: 'Fiber Injection', icon: Zap, fields: [{ name: 'modes', type: 'number', label: 'Modes', step: 1 }] };
                            else if (type === 'fiber_out') data = { ...data, label: 'Fiber Output', icon: Zap, fields: [] };
                            else if (type === 'photonic') data = { ...data, label: 'Photonic Chip', icon: Cpu, fields: [{ name: 'type', type: 'select', label: 'Component Type', options: [{ value: 'y_splitter', label: 'Y-Splitter' }, { value: 'tops', label: 'Phase Shifter' }, { value: 'mmi', label: 'MMI Coupler' }, { value: 'swap', label: 'Waveguide Crossing' }] }, { name: 'phase', type: 'number', label: 'Phase (rad)', step: 0.1 }] };
                        }

                        newNodes.push({ id, type, position, data });

                        if (parentId) {
                            const edge = {
                                id: `e_${parentId}_${id}`,
                                source: parentId,
                                target: id,
                                animated: true
                            };
                            if (layer.metadata && layer.metadata.sourceHandle) {
                                edge.sourceHandle = layer.metadata.sourceHandle;
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

            // Reset file input
            e.target.value = null;

        } catch (e) {
            console.error(e);
            alert("Import Failed: " + e.message);
        }
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

